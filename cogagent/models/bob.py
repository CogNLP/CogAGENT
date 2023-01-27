import torch.nn as nn
import torch
from cogagent.models.modules.bob_encoderdecoder import BobEncoderDecoderModel
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class NLIDataset(Dataset):
    def __init__(self, pre, hyp):
        self.pre = pre
        self.hyp = hyp

    def __getitem__(self, idx):
        pre = {}
        hyp = {}
        pre['input_ids'] = torch.tensor(self.pre[idx][0])
        pre['token_type_ids'] = torch.tensor(self.pre[idx][1])
        pre['attention_mask'] = torch.tensor(self.pre[idx][2])
        hyp['input_ids'] = torch.tensor(self.hyp[idx][0])
        hyp['token_type_ids'] = torch.tensor(self.hyp[idx][1])
        hyp['attention_mask'] = torch.tensor(self.hyp[idx][2])
        return {'pre': pre, 'hyp': hyp}

    def __len__(self):
        return len(self.pre)


class Bob(nn.Module):

    def __init__(self,
                 encoder_model,
                 decoder_model,
                 decoder_model2,
                 nli_batch_size,
                 addition):
        super().__init__()
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.decoder_model2 = decoder_model2
        self.nli_batch_size = nli_batch_size
        self.addition = addition

        self.tokenizer = BertTokenizer.from_pretrained(encoder_model)

        self.encoder_decoder = BobEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=encoder_model,
            decoder_pretrained_model_name_or_path=decoder_model,
            decoder2_pretrained_model_name_or_path=decoder_model2)

        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.tokenizer.eos_token = self.tokenizer.sep_token
        self.encoder_decoder.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.encoder_decoder.config.eos_token_id = self.tokenizer.eos_token_id
        self.encoder_decoder.config.pad_token_id = self.tokenizer.pad_token_id
        self.encoder_decoder.config.vocab_size = self.encoder_decoder.config.decoder.vocab_size
        self.encoder_decoder.config.max_length = 32
        self.encoder_decoder.config.min_length = 3
        self.encoder_decoder.config.no_repeat_ngram_size = 3
        self.encoder_decoder.config.early_stopping = True
        self.encoder_decoder.config.length_penalty = 1.0
        self.encoder_decoder.config.num_beams = 1
        self.encoder_decoder.config.temperature = 0.95
        self.encoder_decoder.config.output_hidden_states = True

        if self.addition is not None:
            postive_nli_dataset = NLIDataset(self.addition['pre_positive'],
                                             self.addition['hyp_positive'])
            negative_nli_dataset = NLIDataset(self.addition['pre_negative'],
                                              self.addition['hyp_negative'])

            self.positive_ul_loader = DataLoader(postive_nli_dataset,
                                                 batch_size=self.nli_batch_size,
                                                 shuffle=True)
            self.negative_ul_loader = DataLoader(negative_nli_dataset,
                                                 batch_size=self.nli_batch_size,
                                                 shuffle=True)
            self.p_ul_iterator = enumerate(self.positive_ul_loader)
            self.p_ul_len = self.positive_ul_loader.__len__()
            self.p_global_step = 0

            self.n_ul_iterator = enumerate(self.negative_ul_loader)
            self.n_ul_len = self.negative_ul_loader.__len__()
            self.n_global_step = 0

    def loss(self, batch, loss_function):
        outputs, outputs_2, ul_outputs = self.forward(batch)
        loss = outputs.loss
        loss_2 = outputs_2.loss
        ul_loss = ul_outputs.loss
        final_loss = loss + 0.01 * loss_2 + 0.01 * ul_loss
        return final_loss

    def forward(self, batch):
        device = batch['input_ids'].device
        input_ids_persona = batch["input_ids_persona"]
        input_ids_query = batch["input_ids_query"]
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        decoder_input_ids_response = batch['decoder_input_ids_response']
        decoder_token_type_ids_response = batch['decoder_token_type_ids_response']
        decoder_attention_mask_response = batch['decoder_attention_mask_response']
        lables = batch['lables']

        if self.p_global_step >= self.p_ul_len - 1:
            self.p_ul_iterator = enumerate(self.positive_ul_loader)
        if self.n_global_step >= self.n_ul_len - 1:
            self.n_ul_iterator = enumerate(self.negative_ul_loader)

        self.p_global_step, pos_batch = next(self.p_ul_iterator)
        self.n_global_step, neg_batch = next(self.n_ul_iterator)

        inference_data_dict = {}
        inference_data_dict['pos_pre_input_ids'] = pos_batch['pre']['input_ids'].to(device)
        inference_data_dict['pos_pre_type_ids'] = pos_batch['pre']['token_type_ids'].to(device)
        inference_data_dict['pos_pre_attention_mask'] = pos_batch['pre']['attention_mask'].to(device)
        inference_data_dict['pos_hyp_input_ids'] = pos_batch['hyp']['input_ids'].to(device)
        inference_data_dict['pos_hyp_type_ids'] = pos_batch['hyp']['token_type_ids'].to(device)
        inference_data_dict['pos_hyp_attention_mask'] = pos_batch['hyp']['attention_mask'].to(device)
        inference_data_dict['neg_pre_input_ids'] = neg_batch['pre']['input_ids'].to(device)
        inference_data_dict['neg_pre_type_ids'] = neg_batch['pre']['token_type_ids'].to(device)
        inference_data_dict['neg_pre_attention_mask'] = neg_batch['pre']['attention_mask'].to(device)
        inference_data_dict['neg_hyp_input_ids'] = neg_batch['hyp']['input_ids'].to(device)
        inference_data_dict['neg_hyp_type_ids'] = neg_batch['hyp']['token_type_ids'].to(device)
        inference_data_dict['neg_hyp_attention_mask'] = neg_batch['hyp']['attention_mask'].to(device)

        outputs, outputs_2, ul_outputs = self.encoder_decoder(input_ids=input_ids,
                                                              attention_mask=attention_mask,
                                                              decoder_input_ids=decoder_input_ids_response,
                                                              decoder_attention_mask=decoder_attention_mask_response,
                                                              labels=lables,
                                                              token_type_ids=token_type_ids,
                                                              training=True,
                                                              return_dict=True,
                                                              per_input_ids=input_ids_persona,
                                                              ul_training=True,
                                                              inference_dict=inference_data_dict,
                                                              )

        return outputs, outputs_2, ul_outputs

    def evaluate(self, batch, metric_function):
        pred = self.predict(batch)

        device = batch['input_ids'].device
        batch_size = len(batch['input_ids'])
        input_ids_persona = batch["input_ids_persona"]
        input_ids_query = batch["input_ids_query"]
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        decoder_input_ids_response = batch['decoder_input_ids_response']
        decoder_token_type_ids_response = batch['decoder_token_type_ids_response']
        decoder_attention_mask_response = batch['decoder_attention_mask_response']
        lables = batch['lables']
        with torch.no_grad():
            outputs_1, outputs_2, _ = self.encoder_decoder(input_ids=input_ids,
                                                           attention_mask=attention_mask,
                                                           decoder_input_ids=decoder_input_ids_response,
                                                           decoder_attention_mask=decoder_attention_mask_response,
                                                           labels=lables,
                                                           token_type_ids=token_type_ids,
                                                           training=True,
                                                           return_dict=True,
                                                           per_input_ids=input_ids_persona,
                                                           ul_training=False,
                                                           inference_dict=None,
                                                           )
            trg_len = decoder_attention_mask_response.sum()
            trg_len_2 = decoder_attention_mask_response.sum()
            log_likelihood_1 = outputs_1.loss * trg_len
            log_likelihood_2 = outputs_2.loss * trg_len_2

            pred["ntokens"].append(trg_len)
            pred["ntokens_2"].append(trg_len_2)
            pred["tokens_loss_1"].append(log_likelihood_1)
            pred["tokens_loss_2"].append(log_likelihood_2)
            pred["sents_loss_1"].append(torch.exp(outputs_1.loss).cpu().item())
            pred["sents_loss_2"].append(torch.exp(outputs_2.loss).cpu().item())

        metric_function.evaluate(pred)

    def predict(self, batch):
        pred = {}
        pred["tokens_loss_1"] = []
        pred["tokens_loss_2"] = []
        pred["ntokens"] = []
        pred["ntokens_2"] = []
        pred["sents_loss_1"] = []
        pred["sents_loss_2"] = []
        pred["example"] = []

        device = batch['input_ids'].device
        batch_size = len(batch['input_ids'])
        input_ids_persona = batch["input_ids_persona"]
        input_ids_query = batch["input_ids_query"]
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        decoder_input_ids_response = batch['decoder_input_ids_response']
        decoder_token_type_ids_response = batch['decoder_token_type_ids_response']
        decoder_attention_mask_response = batch['decoder_attention_mask_response']
        lables = batch['lables']
        with torch.no_grad():
            generated = self.encoder_decoder.generate(input_ids,
                                                      token_type_ids=token_type_ids,
                                                      attention_mask=attention_mask,
                                                      per_input_ids=input_ids_persona)
            generated_2 = self.encoder_decoder.generate(input_ids,
                                                        token_type_ids=token_type_ids,
                                                        attention_mask=attention_mask,
                                                        use_decoder2=True,
                                                        per_input_ids=input_ids_persona)

            generated_token = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            generated_token_2 = self.tokenizer.batch_decode(generated_2, skip_special_tokens=True)
            query_token = self.tokenizer.batch_decode(input_ids_query, skip_special_tokens=True)
            gold_token = self.tokenizer.batch_decode(decoder_input_ids_response, skip_special_tokens=True)
            persona_token = self.tokenizer.batch_decode(input_ids_persona, skip_special_tokens=True)

            for i in range(batch_size):
                example_dict = {}
                example_dict["generated_token"] = generated_token[i]
                example_dict["generated_token_2"] = generated_token_2[i]
                example_dict["query_token"] = query_token[i]
                example_dict["gold_token"] = gold_token[i]
                example_dict["persona_token"] = persona_token[i]
                pred["example"].append(example_dict)

        return pred
