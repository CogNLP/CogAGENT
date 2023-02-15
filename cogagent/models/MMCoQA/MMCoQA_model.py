import os
import logging
import collections

import numpy as np
import scipy as sp
import torch
from transformers import BertModel, BertPreTrainedModel, AlbertModel
from transformers.modeling_bert import (BertEncoder, BertOutput, BertAttention,
                                        BertIntermediate, BertLayer, BertEmbeddings,
                                        BertPooler, BertLayerNorm)
from transformers.modeling_albert import AlbertPreTrainedModel

# from cogagent.data.processors.MMCoQA_processors.MMCoQA_processor import gen_query_reps, retrieve
from cogagent.utils.MMCoQA_utils import (LazyQuacDatasetGlobal, RawResult,
                                         write_predictions, write_final_predictions,
                                         get_retrieval_metrics, gen_reader_features)
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from copy import deepcopy

from transformers.configuration_utils import PretrainedConfig
from transformers.file_utils import (TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, WEIGHTS_NAME,
                                     cached_path)
import torchvision
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


class BertForOrconvqaGlobal(BertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_blocks,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_blocks,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **retrieval_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_blocks,)``:
            Whether the retrieved evidence is the true evidence. For computing the sentece classification loss.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.


    """

    def __init__(self, config):
        super(BertForOrconvqaGlobal, self).__init__(config)
        self.num_qa_labels = config.num_qa_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_qa_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.image_encoder = torchvision.models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, config.hidden_size)

        self.qa_loss_factor = config.qa_loss_factor
        self.retrieval_loss_factor = config.retrieval_loss_factor

        self.query_encoder = BertModel(config)
        self.query_proj = nn.Linear(config.hidden_size, config.proj_size)
        self.modality_detection = nn.Linear(config.proj_size, 3)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                start_positions=None, end_positions=None, retrieval_label=None, image_input=None, modality_labels=None,
                item_modality_type=None,
                query_input_ids=None, query_attention_mask=None, query_token_type_ids=None):

        batch_size, num_blocks, seq_len = input_ids.size()
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        token_type_ids = token_type_ids.view(-1, seq_len)

        # modality_labels = modality_labels.contiguous().view(-1, 1).expand(-1, num_blocks).contiguous().view(-1)

        item_modality_type = item_modality_type.view(-1, 1)

        image_inputs = image_input.view(-1, 3, 512, 512)
        image_inputs = image_inputs.type(torch.FloatTensor)
        image_inputs = image_inputs.to(device)
        image_rep = self.image_encoder(image_inputs)  # (batch_size * num_blocks, hidden_size)
        #        image_rep=self.image_encoder(image_inputs) #(batch_size * num_blocks, hidden_size)
        image_rep = image_rep.view(-1, 1, image_rep.size()[1])

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        #
        sequence_output = sequence_output + image_rep

        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (batch_size * num_blocks, seq_len)
        # print('start_logits', start_logits.size())
        end_logits = end_logits.squeeze(-1)

        pooled_output = self.dropout(pooled_output)
        retrieval_logits = self.classifier(pooled_output)  # (batch_size * num_blocks, 1)

        ###modality detection
        query_outputs = self.query_encoder(query_input_ids,
                                           attention_mask=query_attention_mask,
                                           token_type_ids=query_token_type_ids)

        query_pooled_output = query_outputs[1]
        query_pooled_output = self.dropout(query_pooled_output)
        query_reps = self.query_proj(query_pooled_output)
        query_reps = query_reps.contiguous().view(-1, 1, query_reps.shape[1]).expand(-1, num_blocks, query_reps.shape[
            1]).contiguous().view(-1, query_reps.shape[1])

        modality_logits = self.modality_detection(query_reps)  # (batch_size * num_blocks, 3)

        outputs = (start_logits, end_logits,
                   retrieval_logits + torch.gather(modality_logits, 1, item_modality_type),) + outputs[2:]
        if start_positions is not None and end_positions is not None and retrieval_label is not None:

            modality_labels = modality_labels.contiguous().view(-1, 1).expand(-1, num_blocks).contiguous().view(-1)
            start_logits = start_logits.view(batch_size, -1)
            end_logits = end_logits.view(batch_size, -1)

            retrival_logits = retrieval_logits.squeeze(-1)
            retrieval_logits = retrieval_logits.view(batch_size, -1)

            start_positions = start_positions.squeeze(-1).max(dim=1).values
            end_positions = end_positions.squeeze(-1).max(dim=1).values

            retrieval_label = retrieval_label.squeeze(-1).argmax(dim=1)

            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            qa_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = qa_loss_fct(start_logits, start_positions)
            end_loss = qa_loss_fct(end_logits, end_positions)
            qa_loss = (start_loss + end_loss) / 2

            retrieval_loss_fct = CrossEntropyLoss()

            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)

            modality_loss_fct = CrossEntropyLoss()
            modality_loss = modality_loss_fct(modality_logits, modality_labels)

            total_loss = self.qa_loss_factor * qa_loss + self.retrieval_loss_factor * retrieval_loss + self.retrieval_loss_factor * modality_loss

            outputs = (total_loss, qa_loss, retrieval_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class BertForRetriever(BertPreTrainedModel):
    r"""

    """

    def __init__(self, config):
        super(BertForRetriever, self).__init__(config)

        self.query_encoder = BertModel(config)
        self.query_proj = nn.Linear(config.hidden_size, config.proj_size)

        self.passage_encoder = BertModel(config)
        self.passage_proj = nn.Linear(config.hidden_size, config.proj_size)
        self.proj_size = config.proj_size

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, query_input_ids=None, query_attention_mask=None, query_token_type_ids=None,
                passage_input_ids=None, passage_attention_mask=None, passage_token_type_ids=None,
                retrieval_label=None):
        outputs = ()

        if query_input_ids is not None:
            query_outputs = self.query_encoder(query_input_ids,
                                               attention_mask=query_attention_mask,
                                               token_type_ids=query_token_type_ids)

            query_pooled_output = query_outputs[1]
            query_pooled_output = self.dropout(query_pooled_output)
            query_rep = self.query_proj(query_pooled_output)  # batch_size, proj_size
            # print(query_rep[:, 0])
            outputs = (query_rep,) + outputs

        if passage_input_ids is not None:
            if len(passage_input_ids.size()) == 3:
                # this means we are pretraining
                batch_size, num_blocks, seq_len = passage_input_ids.size()
                passage_input_ids = passage_input_ids.view(-1, seq_len)  # batch_size * num_blocks, seq_len
                passage_attention_mask = passage_attention_mask.view(-1, seq_len)
                passage_token_type_ids = passage_token_type_ids.view(-1, seq_len)

            passage_outputs = self.passage_encoder(passage_input_ids,
                                                   attention_mask=passage_attention_mask,
                                                   token_type_ids=passage_token_type_ids)

            passage_pooled_output = passage_outputs[1]
            passage_pooled_output = self.dropout(passage_pooled_output)
            passage_rep = self.passage_proj(passage_pooled_output)  # batch_size * num_blocks, proj_size
            # print(passage_rep[:, 0])
            outputs = (passage_rep,) + outputs

        if query_input_ids is not None and passage_input_ids is not None and retrieval_label is not None:
            passage_rep = passage_rep.view(batch_size, num_blocks, -1)  # batch_size, num_blocks, proj_size
            query_rep = query_rep.unsqueeze(-1)  # query_rep (batch_size, proj_size, 1)
            query_rep = query_rep.expand(batch_size, self.proj_size, num_blocks)  # batch_size, proj_size, num_blocks)
            query_rep = query_rep.transpose(1, 2)  # query_rep (batch_size, num_blocks, proj_size)
            retrieval_logits = query_rep * passage_rep  # batch_size, num_blocks, proj_size
            retrieval_logits = torch.sum(retrieval_logits, dim=-1)  # batch_size, num_blocks
            retrieval_probs = F.softmax(retrieval_logits, dim=1)
            # print('retrieval_label before', retrieval_label.size(), retrieval_label)
            retrieval_label = retrieval_label.squeeze(-1).argmax(dim=1)
            # print('retrieval_label after', retrieval_label.size(), retrieval_label)
            retrieval_loss_fct = CrossEntropyLoss()
            # print('retrieval_logits', retrieval_logits.size(), retrieval_logits)
            # print('retrieval_label', retrieval_label.size(), retrieval_label)
            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)

            retrieval_logits = retrieval_logits.view(-1)
            outputs = (retrieval_loss, retrieval_logits, retrieval_probs) + outputs

        return outputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        """
        if pretrained_model_name_or_path is not None and (
                "albert" in pretrained_model_name_or_path and "v2" in pretrained_model_name_or_path):
            logger.warning("There is currently an upstream reproducibility issue with ALBERT v2 models. Please see " +
                           "https://github.com/google-research/google-research/issues/119 for more information.")

        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        from_tf = kwargs.pop('from_tf', False)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        output_loading_info = kwargs.pop('output_loading_info', False)

        # Load config
        if config is None:
            config, model_kwargs = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, *model_args,
                cache_dir=cache_dir, return_unused_kwargs=True,
                force_download=force_download,
                proxies=proxies,
                **kwargs
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
            elif os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    # Load from a TF 1.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        "Error no file named {} found in directory {} or `from_tf` set to False".format(
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
                            pretrained_model_name_or_path))
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                assert from_tf, "We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
                    pretrained_model_name_or_path + ".index")
                archive_file = pretrained_model_name_or_path + ".index"

            # redirect to the cache, if necessary
            try:
                resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir, force_download=force_download,
                                                    proxies=proxies)
            except EnvironmentError:
                if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                    msg = "Couldn't reach server at '{}' to download pretrained weights.".format(
                        archive_file)
                else:
                    msg = "Model name '{}' was not found in model name list ({}). " \
                          "We assumed '{}' was a path or url to model weight files named one of {} but " \
                          "couldn't find any such file at this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(cls.pretrained_model_archive_map.keys()),
                        archive_file,
                        [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME])
                raise EnvironmentError(msg)

            if resolved_archive_file == archive_file:
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("loading weights file {} from cache at {}".format(
                    archive_file, resolved_archive_file))
        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        if from_tf:
            if resolved_archive_file.endswith('.index'):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from transformers import load_tf2_checkpoint_in_pytorch_model
                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError as e:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.")
                    raise e
        else:
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if 'gamma' in key:
                    new_key = key.replace('gamma', 'weight')
                if 'beta' in key:
                    new_key = key.replace('beta', 'bias')
                if key == 'lm_head.decoder.weight':
                    new_key = 'lm_head.weight'
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, '_metadata', None)
            # print('orig state dict', state_dict.keys(), len(state_dict))
            customized_state_dict = collections.OrderedDict()
            for k, v in state_dict.items():
                k_split = k.split('.')
                if k_split[0] == 'bert':
                    k_split[0] = 'query_encoder'
                    customized_state_dict['.'.join(k_split)] = v
                    k_split[0] = 'passage_encoder'
                    customized_state_dict['.'.join(k_split)] = v

            if len(customized_state_dict) == 0:
                # loading from our trained model
                state_dict = state_dict.copy()
                # print('using orig state dict', state_dict.keys())
            else:
                # loading from original bert model
                state_dict = customized_state_dict.copy()
                # print('using custome state dict', state_dict.keys())

            # print('modified state dict', state_dict.keys(), len(state_dict))
            if metadata is not None:
                state_dict._metadata = metadata

            # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
            # so we need to apply the function recursively.
            def load(module, prefix=''):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + '.')

            # Make sure we are able to load base models as well as derived models (with heads)
            start_prefix = ''
            model_to_load = model
            #  if not hasattr(model, cls.base_model_prefix) and any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            #  start_prefix = cls.base_model_prefix + '.'
            #  if hasattr(model, cls.base_model_prefix) and not any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            #  model_to_load = getattr(model, cls.base_model_prefix)

            #             load(model_to_load, prefix=start_prefix)
            load(model_to_load, prefix='')
            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys))
            if len(error_msgs) > 0:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                    model.__class__.__name__, "\n\t".join(error_msgs)))

        model.tie_weights()  # make sure word embedding weights are still tied if needed

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model


class BertForRetrieverOnlyPositivePassage(BertForRetriever):
    r"""

    """

    def __init__(self, config):
        super(BertForRetriever, self).__init__(config)

        self.query_encoder = BertModel(config)
        config.proj_size = 128
        self.query_proj = nn.Linear(config.hidden_size, config.proj_size)

        self.passage_encoder = BertModel(config)
        self.passage_proj = nn.Linear(config.hidden_size, config.proj_size)
        self.proj_size = config.proj_size

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.image_encoder = torchvision.models.resnet101(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, config.hidden_size)
        self.image_proj = nn.Linear(config.hidden_size, config.proj_size)

        self.init_weights()

    def forward(self, query_input_ids=None, query_attention_mask=None, query_token_type_ids=None,
                passage_input_ids=None, passage_attention_mask=None, passage_token_type_ids=None,
                retrieval_label=None, question_type=None, image_input=None, query_rep=None, passage_rep=None,
                modality_labels=None):
        outputs = ()

        if query_input_ids is not None:
            query_outputs = self.query_encoder(query_input_ids,
                                               attention_mask=query_attention_mask,
                                               token_type_ids=query_token_type_ids)

            query_pooled_output = query_outputs[1]
            query_pooled_output = self.dropout(query_pooled_output)
            query_rep = self.query_proj(query_pooled_output)  # batch_size, proj_size
            # print(query_rep[:, 0])
            outputs = (query_rep,) + outputs

        if passage_input_ids is not None:
            passage_outputs = self.passage_encoder(passage_input_ids,
                                                   attention_mask=passage_attention_mask,
                                                   token_type_ids=passage_token_type_ids)

            passage_pooled_output = passage_outputs[1]
            passage_pooled_output = self.dropout(passage_pooled_output)
            passage_rep = self.passage_proj(passage_pooled_output)  # batch_size, proj_size
            # print(passage_rep[:, 0])

            #####################encode an image
            image_outputs = self.image_encoder(image_input)
            image_rep = self.image_proj(image_outputs)  # batch_size, proj_size

            ##############obtain the corresponding embedding     modality_position=question_type:[0,1,0,1]*batchsize+[0,1,2,3]=[0,4,2,6]

            modality_position = question_type * passage_rep.size(0) + torch.arange(passage_rep.size(0),
                                                                                   device=passage_rep.device,
                                                                                   dtype=torch.long)

            passage_rep = torch.cat((passage_rep, image_rep), 0)[modality_position]

            outputs = (passage_rep,) + outputs

        if query_input_ids is not None and passage_input_ids is not None:
            passage_rep_t = passage_rep.transpose(0, 1)  # proj_size, batch_size
            retrieval_logits = torch.matmul(query_rep, passage_rep_t)  # batch_size, batch_size
            retrieval_label = torch.arange(query_rep.size(0), device=query_rep.device, dtype=retrieval_label.dtype)
            # print('retrieval_label after', retrieval_label.size(), retrieval_label)
            retrieval_loss_fct = CrossEntropyLoss()
            # print('retrieval_logits', retrieval_logits.size(), retrieval_logits)
            # print('retrieval_label', retrieval_label.size(), retrieval_label)
            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)

            outputs = (retrieval_loss,) + outputs

        if query_input_ids is not None and passage_rep is not None and retrieval_label is not None and len(
                passage_rep.size()) == 3:
            # this is during fine tuning
            # passage_rep: batch_size, num_blocks, proj_size
            query_outputs = self.query_encoder(query_input_ids,
                                               attention_mask=query_attention_mask,
                                               token_type_ids=query_token_type_ids)

            query_pooled_output = query_outputs[1]
            query_pooled_output = self.dropout(query_pooled_output)
            query_rep = self.query_proj(query_pooled_output)  # batch_size, proj_size

            batch_size, num_blocks, proj_size = passage_rep.size()
            query_rep = query_rep.unsqueeze(-1)  # query_rep (batch_size, proj_size, 1)
            query_rep = query_rep.expand(batch_size, self.proj_size, num_blocks)  # batch_size, proj_size, num_blocks)
            query_rep = query_rep.transpose(1, 2)  # query_rep (batch_size, num_blocks, proj_size)
            retrieval_logits = query_rep * passage_rep  # batch_size, num_blocks, proj_size
            retrieval_logits = torch.sum(retrieval_logits, dim=-1)  # batch_size, num_blocks
            retrieval_probs = F.softmax(retrieval_logits, dim=1)
            # print('retrieval_label before', retrieval_label.size(), retrieval_label)
            retrieval_label = retrieval_label.squeeze(-1).argmax(dim=1)
            # print('retrieval_label after', retrieval_label.size(), retrieval_label)
            retrieval_loss_fct = CrossEntropyLoss()
            # print('retrieval_logits', retrieval_logits.size(), retrieval_logits)
            # print('retrieval_label', retrieval_label.size(), retrieval_label)
            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)

            outputs = (retrieval_loss,) + outputs

        if query_input_ids is not None and modality_labels is not None:
            # this is during fine tuning
            # passage_rep: batch_size, num_blocks, proj_size
            query_outputs = self.query_encoder(query_input_ids,
                                               attention_mask=query_attention_mask,
                                               token_type_ids=query_token_type_ids)

            query_pooled_output = query_outputs[1]
            query_pooled_output = self.dropout(query_pooled_output)
            query_rep = self.query_proj(query_pooled_output)  # batch_size, proj_size

            outputs = (retrieval_loss,) + outputs
        return outputs


class Pipeline(nn.Module):
    def __init__(self, reader_tokenizer):
        super(Pipeline, self).__init__()

        self.reader = None
        self.retriever = None
        self.reader_tokenizer = reader_tokenizer
        self.top_k_for_retriever = 2000
        self.top_k_for_reader = 5
        self.reader_max_seq_length = 512


    def loss(self, batch, loss_function):
        query_input_ids = batch["train_query_input_ids"]
        query_attention_mask = batch["train_query_attention_mask"]
        query_token_type_ids = batch["train_query_token_type_ids"]
        qids = batch["qids"]
        question_texts = []
        answer_texts = []
        answer_starts = []

        for i in range(len(batch["question_texts"])):
            question_texts.append(batch["question_texts"][i][0])
        for i in range(len(batch["answer_texts"])):
            answer_texts.append(batch["answer_texts"][i][0])
        for i in range(len(batch["answer_starts"])):
            answer_starts.append(0)

        reps_inputs = {'query_input_ids': query_input_ids,
                       'query_attention_mask': query_attention_mask,
                       'query_token_type_ids': query_token_type_ids}
        query_reps = self.gen_query_reps(reps_inputs)

        qid_to_idx = batch["qid_to_idx"][0]
        item_ids = batch["item_ids"][0]
        item_id_to_idx = batch["item_id_to_idx"][0]
        item_reps = batch["item_reps"][0]
        qrels = batch["qrels"][0]
        qrels_sparse_matrix = batch["qrels_sparse_matrix"][0]
        gpu_index = batch["gpu_index"][0]
        itemid_modalities = batch["itemid_modalities"][0]
        images_titles = batch["images_titles"][0]


        get_passage_batch = {'itemid_modalities': batch["itemid_modalities"][0],
                             'passages_dict': batch["passages_dict"][0],
                             'tables_dict': batch["tables_dict"][0],
                             'images_dict': batch["images_dict"][0],
                             'item_ids': batch["item_ids"][0]}

        retrieval_results = self.retrieve(self.top_k_for_retriever, self.top_k_for_reader, qids, qid_to_idx,
                                     query_reps, item_ids, item_id_to_idx, item_reps, qrels, qrels_sparse_matrix,
                                     gpu_index, get_passage_batch, include_positive_passage=True)
        passage_reps_for_retriever = retrieval_results['passage_reps_for_retriever']  # (batch,2000,128) float32
        labels_for_retriever = retrieval_results['labels_for_retriever']  # (batch,2000) int64
        pids_for_reader = retrieval_results['pids_for_reader']  # (batch,5)
        passages_for_reader = retrieval_results['passages_for_reader']  # (batch,5)
        labels_for_reader = retrieval_results['labels_for_reader']  # (batch,5)
        # reader的数据
        reader_batch = gen_reader_features(qids, question_texts, answer_texts, answer_starts,
                                           pids_for_reader, passages_for_reader, labels_for_reader,
                                           self.reader_tokenizer, self.reader_max_seq_length, is_training=True,
                                           itemid_modalities=itemid_modalities,
                                           item_id_to_idx=item_id_to_idx,
                                           images_titles=images_titles)


        passage_rep = torch.from_numpy(passage_reps_for_retriever)
        retriever_label = torch.from_numpy(labels_for_retriever)
        inputs = {'query_input_ids': batch['train_query_input_ids'].to(device),
                  'query_attention_mask': batch['train_query_attention_mask'].to(device),
                  'query_token_type_ids': batch['train_query_token_type_ids'].to(device),
                  'passage_rep': passage_rep.to(device),
                  'retrieval_label': retriever_label.to(device)}

        retriever_outputs = self.retriever(**inputs)
        retriever_loss = retriever_outputs[0]

        inputs = {'input_ids': reader_batch['input_ids'].to(device),
                  'attention_mask': reader_batch['input_mask'].to(device),
                  'token_type_ids': reader_batch['segment_ids'].to(device),
                  'start_positions': reader_batch['start_position'].to(device),
                  'end_positions': reader_batch['end_position'].to(device),
                  'retrieval_label': reader_batch['retrieval_label'].to(device),
                  'image_input': reader_batch['image_input'].to(device),
                  'modality_labels': batch['train_modality_labels'].to(device),
                  'item_modality_type': reader_batch['item_modality_type'].to(device),
                  'query_input_ids': batch['train_query_input_ids'].to(device),
                  'query_attention_mask': batch['train_query_attention_mask'].to(device),
                  'query_token_type_ids': batch['train_query_token_type_ids'].to(device)}
        reader_outputs = self.reader(**inputs)

        reader_loss, qa_loss, rerank_loss = reader_outputs[0:3]
        loss = retriever_loss + reader_loss

        return loss

    def evaluate(self, batch, metric_function):
        all_results, examples, features, retriever_run_dict = self.predict(batch)
        metric_function.evaluate(all_results, examples, features, retriever_run_dict)



    def predict(self, batch):
        query_input_ids = batch["dev_query_input_ids"]
        query_attention_mask = batch["dev_query_attention_mask"]
        query_token_type_ids = batch["dev_query_token_type_ids"]
        qids = batch["qids"]
        question_texts = []
        answer_texts = []
        answer_starts = []

        for i in range(len(batch["question_texts"])):
            question_texts.append(batch["question_texts"][i][0])
        for i in range(len(batch["answer_texts"])):
            answer_texts.append(batch["answer_texts"][i][0])
        for i in range(len(batch["answer_starts"])):
            answer_starts.append(0)

        reps_inputs = {'query_input_ids': query_input_ids,
                       'query_attention_mask': query_attention_mask,
                       'query_token_type_ids': query_token_type_ids}
        query_reps = self.gen_query_reps(reps_inputs)

        qid_to_idx = batch["qid_to_idx"][0]
        item_ids = batch["item_ids"][0]
        item_id_to_idx = batch["item_id_to_idx"][0]
        item_reps = batch["item_reps"][0]
        qrels = batch["qrels"][0]
        qrels_sparse_matrix = batch["qrels_sparse_matrix"][0]
        gpu_index = batch["gpu_index"][0]
        itemid_modalities = batch["itemid_modalities"][0]
        images_titles = batch["images_titles"][0]

        get_passage_batch = {'itemid_modalities': batch["itemid_modalities"][0],
                             'passages_dict': batch["passages_dict"][0],
                             'tables_dict': batch["tables_dict"][0],
                             'images_dict': batch["images_dict"][0],
                             'item_ids': batch["item_ids"][0]}
        #
        retrieval_results = self.retrieve(self.top_k_for_retriever, self.top_k_for_reader, qids, qid_to_idx,
                                      query_reps, item_ids, item_id_to_idx, item_reps, qrels, qrels_sparse_matrix,
                                      gpu_index, get_passage_batch, include_positive_passage=True)
        pids_for_retriever = retrieval_results['pids_for_retriever']
        retriever_probs = retrieval_results['retriever_probs']
        retriever_run_dict, rarank_run_dict = {}, {}
        examples, features = {}, {}
        for i in range(len(qids)):
            retriever_run_dict[qids[i]] = {}
            for j in range(retrieval_results['no_cut_retriever_probs'].shape[1]):
                retriever_run_dict[qids[i]][pids_for_retriever[i, j]] = int(
                    retrieval_results['no_cut_retriever_probs'][i, j])
        pids_for_reader = retrieval_results['pids_for_reader']  # (1,5)
        passages_for_reader = retrieval_results['passages_for_reader']
        labels_for_reader = retrieval_results['labels_for_reader']

        # reader的数据
        reader_batch, batch_examples, batch_features = gen_reader_features(qids, question_texts, answer_texts,
                                                                           answer_starts,
                                                                           pids_for_reader, passages_for_reader,
                                                                           labels_for_reader,
                                                                           self.reader_tokenizer,
                                                                           self.reader_max_seq_length,
                                                                           is_training=False,
                                                                           itemid_modalities=itemid_modalities,
                                                                           item_id_to_idx=item_id_to_idx,
                                                                           images_titles=images_titles)
        example_ids = reader_batch['example_id']
        examples.update(batch_examples)
        features.update(batch_features)
        reader_batch = {k: v.to(device) for k, v in reader_batch.items() if k != 'example_id'}
        # retriever_probs = retriever_probs[0]
        retriever_probs = retriever_probs.reshape(-1).tolist()


        inputs = {'input_ids': reader_batch['input_ids'].to(device),
                  'attention_mask': reader_batch['input_mask'].to(device),
                  'token_type_ids': reader_batch['segment_ids'].to(device),
                  'image_input': reader_batch['image_input'].to(device),
                  'modality_labels': batch['dev_modality_labels'].to(device),
                  'item_modality_type': reader_batch['item_modality_type'].to(device),
                  'query_input_ids': batch['dev_query_input_ids'].to(device),
                  'query_attention_mask': batch['dev_query_attention_mask'].to(device),
                  'query_token_type_ids': batch['dev_query_token_type_ids'].to(device)}
        outputs = self.reader(**inputs)
        # example_ids = []
        # examples = {}
        # features = {}
        #
        # for i in range(len(dev_example_ids)):
        #     dev_example_ids_i = dev_example_ids[i]
        #     for j in range(len(dev_example_ids_i)):
        #         example_ids.append(dev_example_ids_i[j])
        # for i in range(len(dev_examples)):
        #     examples.update(dev_examples[i])
        # for i in range(len(dev_features)):
        #     features.update(dev_features[i])
        # retriever_probs = batch['dev_retriever_probs']
        # retriever_probs = retriever_probs[0].tolist()
        all_results = []
        for i, example_id in enumerate(example_ids):
            result = RawResult(unique_id=example_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]),
                               retrieval_logits=to_list(outputs[2][i]),
                               retriever_prob=retriever_probs[i])

            all_results.append(result)
        return all_results, examples, features, retriever_run_dict

    def gen_query_reps(self, batch):
        # model.eval()
        batch = {k: v.to("cuda:6") for k, v in batch.items()
                 if k not in ['example_id', 'qid', 'question_text', 'answer_text', 'answer_start']}
        with torch.no_grad():
            # model = model.to("cuda:0")
            inputs = {}
            inputs['query_input_ids'] = batch['query_input_ids']
            inputs['query_attention_mask'] = batch['query_attention_mask']
            inputs['query_token_type_ids'] = batch['query_token_type_ids']
            outputs = self.retriever(**inputs)
            query_reps = outputs[0]

        return query_reps

    def retrieve(self, top_k_for_retriever, top_k_for_reader, qids, qid_to_idx, query_reps,
                 item_ids, item_id_to_idx, item_reps, qrels, qrels_sparse_matrix,
                 gpu_index, batch, include_positive_passage=False):
        query_reps = query_reps.detach().cpu().numpy()
        D, I = gpu_index.search(query_reps, top_k_for_retriever)

        pidx_for_retriever = np.copy(I)
        qidx = [qid_to_idx[qid] for qid in qids]
        qidx_expanded = np.expand_dims(qidx, axis=1)
        qidx_expanded = np.repeat(qidx_expanded, top_k_for_retriever, axis=1)
        labels_for_retriever = qrels_sparse_matrix[qidx_expanded, pidx_for_retriever].toarray()
        if include_positive_passage:
            for i, (qid, labels_per_query) in enumerate(zip(qids, labels_for_retriever)):
                has_positive = np.sum(labels_per_query)
                if not has_positive:
                    positive_pid = list(qrels[qid].keys())[0]
                    positive_pidx = item_id_to_idx[positive_pid]
                    pidx_for_retriever[i][-1] = positive_pidx
            labels_for_retriever = qrels_sparse_matrix[qidx_expanded, pidx_for_retriever].toarray()
            assert np.sum(labels_for_retriever) >= len(labels_for_retriever)
        pids_for_retriever = item_ids[pidx_for_retriever]
        passage_reps_for_retriever = item_reps[pidx_for_retriever]

        scores = D[:, :top_k_for_reader]
        retriever_probs = sp.special.softmax(scores, axis=1)
        pidx_for_reader = I[:, :top_k_for_reader]
        qidx_expanded = np.expand_dims(qidx, axis=1)
        qidx_expanded = np.repeat(qidx_expanded, top_k_for_reader, axis=1)
        labels_for_reader = qrels_sparse_matrix[qidx_expanded, pidx_for_reader].toarray()
        if include_positive_passage:
            for i, (qid, labels_per_query) in enumerate(zip(qids, labels_for_reader)):
                has_positive = np.sum(labels_per_query)
                if not has_positive:
                    positive_pid = list(qrels[qid].keys())[0]
                    positive_pidx = item_id_to_idx[positive_pid]
                    pidx_for_reader[i][-1] = positive_pidx
            labels_for_reader = qrels_sparse_matrix[qidx_expanded, pidx_for_reader].toarray()
            assert np.sum(labels_for_reader) >= len(labels_for_reader)
        pids_for_reader = item_ids[pidx_for_reader]
        passages_for_reader = get_passages(pidx_for_reader, batch)
        if passages_for_reader.size == 5:
            passages_for_reader = np.expand_dims(passages_for_reader, 0)  # 如果维度是5，原来size 5 增加一个维度 变成（1,5）


        return {'qidx': qidx,
                'pidx_for_retriever': pidx_for_retriever,
                'pids_for_retriever': pids_for_retriever,  #
                'passage_reps_for_retriever': passage_reps_for_retriever,
                'labels_for_retriever': labels_for_retriever,
                'retriever_probs': retriever_probs,  #
                'pidx_for_reader': pidx_for_reader,
                'pids_for_reader': pids_for_reader,  #
                'passages_for_reader': passages_for_reader,  #
                'labels_for_reader': labels_for_reader,  #
                'no_cut_retriever_probs': D}  #

    def tookit_retrieve(self, top_k_for_retriever, top_k_for_reader, qids, qid_to_idx, query_reps,
                 item_ids, item_id_to_idx, item_reps, qrels, qrels_sparse_matrix,
                 gpu_index, batch, include_positive_passage=False):
        query_reps = query_reps.detach().cpu().numpy()
        D, I = gpu_index.search(query_reps, top_k_for_retriever)

        pidx_for_retriever = np.copy(I)
        qidx = [qid_to_idx[qid] for qid in qids]
        # qidx_expanded = np.expand_dims(qidx, axis=1)
        # qidx_expanded = np.repeat(qidx_expanded, top_k_for_retriever, axis=1)
        # labels_for_retriever = qrels_sparse_matrix[qidx_expanded, pidx_for_retriever].toarray()
        # if include_positive_passage:
        #     for i, (qid, labels_per_query) in enumerate(zip(qids, labels_for_retriever)):
        #         has_positive = np.sum(labels_per_query)
        #         if not has_positive:
        #             positive_pid = list(qrels[qid].keys())[0]
        #             positive_pidx = item_id_to_idx[positive_pid]
        #             pidx_for_retriever[i][-1] = positive_pidx
        #     labels_for_retriever = qrels_sparse_matrix[qidx_expanded, pidx_for_retriever].toarray()
        #     assert np.sum(labels_for_retriever) >= len(labels_for_retriever)
        pids_for_retriever = item_ids[pidx_for_retriever]
        passage_reps_for_retriever = item_reps[pidx_for_retriever]

        scores = D[:, :top_k_for_reader]
        retriever_probs = sp.special.softmax(scores, axis=1)
        pidx_for_reader = I[:, :top_k_for_reader]
        qidx_expanded = np.expand_dims(qidx, axis=1)
        qidx_expanded = np.repeat(qidx_expanded, top_k_for_reader, axis=1)
        labels_for_reader = qrels_sparse_matrix[qidx_expanded, pidx_for_reader].toarray()
        # if include_positive_passage:
        #     for i, (qid, labels_per_query) in enumerate(zip(qids, labels_for_reader)):
        #         has_positive = np.sum(labels_per_query)
        #         if not has_positive:
        #             positive_pid = list(qrels[qid].keys())[0]
        #             positive_pidx = item_id_to_idx[positive_pid]
        #             pidx_for_reader[i][-1] = positive_pidx
        #     labels_for_reader = qrels_sparse_matrix[qidx_expanded, pidx_for_reader].toarray()
        #     assert np.sum(labels_for_reader) >= len(labels_for_reader)
        pids_for_reader = item_ids[pidx_for_reader]
        passages_for_reader = get_passages(pidx_for_reader, batch)
        if passages_for_reader.size == 5:
            passages_for_reader = np.expand_dims(passages_for_reader, 0)  # 如果维度是5，原来size 5 增加一个维度 变成（1,5）


        return {# 'qidx': qidx,
                'pidx_for_retriever': pidx_for_retriever,
                'pids_for_retriever': pids_for_retriever,  #
                'passage_reps_for_retriever': passage_reps_for_retriever,
                # 'labels_for_retriever': labels_for_retriever,
                'retriever_probs': retriever_probs,  #
                # 'pidx_for_reader': pidx_for_reader,
                'pids_for_reader': pids_for_reader,  #
                'passages_for_reader': passages_for_reader,  #
                'labels_for_reader': labels_for_reader,  #
                'no_cut_retriever_probs': D}  #


def to_list(tensor):
    return tensor.detach().cpu().tolist()
def get_passages(ids, batch):
    ids = list(np.squeeze(ids))
    len_ids = len(ids)
    item_contexts = []
    # temp = ids[0]
    # if len_ids == 1:
    #     all_ids = ids
    # if len_ids > 1:
    #     for i in range(len_ids - 1):
    #         temp = np.concatenate((temp, ids[i+1]), axis=0)
    #     all_ids = temp
    if len_ids == 5:
        for i in ids:
            itemid_modalities = batch["itemid_modalities"]
            passages_dict = batch["passages_dict"]
            tables_dict = batch["tables_dict"]
            images_dict = batch["images_dict"]
            item_ids = batch["item_ids"]
            if itemid_modalities[i] == 'text':
                item_context = passages_dict[item_ids[i]]
            elif itemid_modalities[i] == 'table':
                item_context = tables_dict[item_ids[i]]
            elif itemid_modalities[i] == 'image':
                item_context = images_dict[item_ids[i]]
            item_contexts.append(item_context)

    if len_ids < 5:
        for i in range(len_ids):
            temp = ids[i]
            item_context_temp = []
            for j in temp:
                itemid_modalities = batch["itemid_modalities"]
                passages_dict = batch["passages_dict"]
                tables_dict = batch["tables_dict"]
                images_dict = batch["images_dict"]
                item_ids = batch["item_ids"]
                if itemid_modalities[j] == 'text':
                    item_context = passages_dict[item_ids[j]]
                elif itemid_modalities[j] == 'table':
                    item_context = tables_dict[item_ids[j]]
                elif itemid_modalities[j] == 'image':
                    item_context = images_dict[item_ids[j]]
                item_context_temp.append(item_context)
            item_contexts.append(item_context_temp)

    return np.array(item_contexts)