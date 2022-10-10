from doctest import OutputChecker
import json
import os
import torch
from torch import nn
from transformers import BertModel
from torch.utils.tensorboard import SummaryWriter
# from torch.nn.parameter import Parameter
from cogagent.models.base_model import BaseModel
from cogagent.data.processors.jointbert_processors.postprocess import is_slot_da, calculateF1, recover_intent
class JointbertModel(BaseModel):
    def __init__(self, model_config, device, slot_dim, intent_dim, intent_weight=None):
        super(JointbertModel, self).__init__()
        self.slot_num_labels = slot_dim
        self.intent_num_labels = intent_dim
        self.device = device
        self.intent_weight = intent_weight if intent_weight is not None else torch.tensor([1.]*intent_dim)

        print(model_config['pretrained_weights'])
        self.bert = BertModel.from_pretrained(model_config['pretrained_weights'])
        self.dropout = nn.Dropout(model_config['dropout'])
        self.context = model_config['context']
        self.finetune = model_config['finetune']
        self.context_grad = model_config['context_grad']
        self.hidden_units = model_config['hidden_units']
        if self.hidden_units > 0:
            if self.context:
                self.intent_classifier = nn.Linear(self.hidden_units, self.intent_num_labels)
                self.slot_classifier = nn.Linear(self.hidden_units, self.slot_num_labels)
                self.intent_hidden = nn.Linear(2 * self.bert.config.hidden_size, self.hidden_units)
                self.slot_hidden = nn.Linear(2 * self.bert.config.hidden_size, self.hidden_units)
            else:
                self.intent_classifier = nn.Linear(self.hidden_units, self.intent_num_labels)
                self.slot_classifier = nn.Linear(self.hidden_units, self.slot_num_labels)
                self.intent_hidden = nn.Linear(self.bert.config.hidden_size, self.hidden_units)
                self.slot_hidden = nn.Linear(self.bert.config.hidden_size, self.hidden_units)
            nn.init.xavier_uniform_(self.intent_hidden.weight)
            nn.init.xavier_uniform_(self.slot_hidden.weight)
        else:
            if self.context:
                self.intent_classifier = nn.Linear(2 * self.bert.config.hidden_size, self.intent_num_labels)
                self.slot_classifier = nn.Linear(2 * self.bert.config.hidden_size, self.slot_num_labels)
            else:
                self.intent_classifier = nn.Linear(self.bert.config.hidden_size, self.intent_num_labels)
                self.slot_classifier = nn.Linear(self.bert.config.hidden_size, self.slot_num_labels)
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.xavier_uniform_(self.slot_classifier.weight)

        self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)
        self.slot_loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, word_seq_tensor, word_mask_tensor, tag_seq_tensor=None, tag_mask_tensor=None,
                intent_tensor=None, context_seq_tensor=None, context_mask_tensor=None):
        if not self.finetune:
            self.bert.eval()
            with torch.no_grad():
                outputs = self.bert(input_ids=word_seq_tensor,
                                    attention_mask=word_mask_tensor)
        else:
            outputs = self.bert(input_ids=word_seq_tensor,
                                attention_mask=word_mask_tensor)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        if self.context and (context_seq_tensor is not None):
            if not self.finetune or not self.context_grad:
                with torch.no_grad():
                    context_output = self.bert(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            else:
                context_output = self.bert(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            sequence_output = torch.cat(
                [context_output.unsqueeze(1).repeat(1, sequence_output.size(1), 1),
                 sequence_output], dim=-1)
            pooled_output = torch.cat([context_output, pooled_output], dim=-1)

        if self.hidden_units > 0:
            sequence_output = nn.functional.relu(self.slot_hidden(self.dropout(sequence_output)))
            pooled_output = nn.functional.relu(self.intent_hidden(self.dropout(pooled_output)))

        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)
        outputs = (slot_logits,)

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        outputs = outputs + (intent_logits,)

        if tag_seq_tensor is not None:
            active_tag_loss = tag_mask_tensor.view(-1) == 1
            active_tag_logits = slot_logits.view(-1, self.slot_num_labels)[active_tag_loss]
            active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
            slot_loss = self.slot_loss_fct(active_tag_logits, active_tag_labels)

            outputs = outputs + (slot_loss,)

        if intent_tensor is not None:
            intent_loss = self.intent_loss_fct(intent_logits, intent_tensor)
            outputs = outputs + (intent_loss,)

        return outputs  # slot_logits, intent_logits, (slot_loss), (intent_loss),


    def loss(self, batch, loss_function):
        word_seq_tensor = batch['word_seq_tensor']
        word_mask_tensor = batch['word_mask_tensor']
        tag_seq_tensor = batch['tag_seq_tensor']
        tag_mask_tensor = batch['tag_mask_tensor']
        intent_tensor = batch['intent_tensor']
        context_seq_tensor = batch['context_seq_tensor']
        context_mask_tensor = batch['context_mask_tensor']
        # slot_logits, intent_logits, slot_loss, intent_loss = self.forward(batch)
        output = ()
        output = self.forward(word_seq_tensor, word_mask_tensor, tag_seq_tensor, tag_mask_tensor, 
                                                                            intent_tensor, 
                                                                            context_seq_tensor, 
                                                                            context_mask_tensor)
        slot_logits = output[0]
        intent_logits = output[1]                                                          
        if tag_seq_tensor is not None:
            active_tag_loss = tag_mask_tensor.view(-1) == 1
            active_tag_logits = slot_logits.view(-1, self.slot_num_labels)[active_tag_loss]
            active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
            slot_loss = loss_function["crossentropyloss"](active_tag_logits, active_tag_labels)

        if intent_tensor is not None:
            intent_loss = loss_function["bcewithlogitsloss"](intent_logits, intent_tensor)
        loss = slot_loss + intent_loss
        return loss


    # def evaluate(self, batch, metric_function):
          
    #     predict_golden = {'intent': [], 'slot': [], 'overall': []}
    #     val_slot_loss, val_intent_loss = 0, 0
    #     for pad_batch, ori_batch, real_batch_size in self.processor.yield_batches(batch_size, data_key='val'):
    #         word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
    #         with torch.no_grad():
    #             slot_logits, intent_logits, slot_loss, intent_loss = self.model.forward(word_seq_tensor,
    #                                                                                     word_mask_tensor,
    #                                                                                     tag_seq_tensor,
    #                                                                                     tag_mask_tensor,
    #                                                                                     intent_tensor,
    #                                                                                     context_seq_tensor,
    #                                                                                     context_mask_tensor)
    #         val_slot_loss += slot_loss.item() * real_batch_size
    #         val_intent_loss += intent_loss.item() * real_batch_size
    #         for j in range(real_batch_size):
    #             predicts = recover_intent(self.processor, intent_logits[j], slot_logits[j], tag_mask_tensor[j],
    #                                             ori_batch[j][0], ori_batch[j][-4])
    #             labels = ori_batch[j][3]

    #             predict_golden['overall'].append({
    #                         'predict': predicts,
    #                         'golden': labels
    #                     })
    #             predict_golden['slot'].append({
    #                         'predict': [x for x in predicts if is_slot_da(x)],
    #                         'golden': [x for x in labels if is_slot_da(x)]
    #                     })
    #             predict_golden['intent'].append({
    #                         'predict': [x for x in predicts if not is_slot_da(x)],
    #                         'golden': [x for x in labels if not is_slot_da(x)]
    #                     })

    #             total = len(self.processor.data['val'][0])
    #             val_slot_loss /= total
    #             val_intent_loss /= total
                
    #             for x in ['intent', 'slot', 'overall']:
    #                 precision, recall, F1 = calculateF1(predict_golden[x])
                    

    #             if F1 > best_val_f1:
    #                 best_val_f1 = F1
    #                 torch.save(self.model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
         
    #     metric_function.evaluate(val_slot_loss, val_intent_loss,precision, recall, F1)

    # def predict(self, batch):
    #     pred = self.forward(batch)
    #     pred = F.softmax(pred, dim=1)
    #     pred = torch.max(pred, dim=1)[1]
    #     return pred