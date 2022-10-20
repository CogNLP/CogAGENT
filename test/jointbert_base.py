import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AdamW, get_linear_schedule_with_warmup
import zipfile
# 解决 ModuleNotFoundError: No module named 'cogagent' 问题
# from cogagent.data.readers.jointbert_reader import JointbertReader
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cogagent.data.processors.jointbert_processors.multiwoz_processor import MultiwozProcessor
from cogagent.core.metric.base_jointbert_metric import BaseJointbertMetric
from cogagent.data import *
from cogagent.data.processors import *
from cogagent.data.processors.jointbert_processors import *
from cogagent.core import *
from cogagent.models import *
from cogagent.toolkits import *
from cogagent.utils import *
# from cogagent.data.processors.jointbert_processors.postprocess import *
from cogagent.data.processors.jointbert_processors.postprocess import is_slot_da, calculateF1, recover_intent

device, output_path = init_cogagent(
    device_id=1,
    output_path="/home/nlp/anaconda3/envs/CogAGENT/CogAGENT/datapath/nlu/multiwoz/experimental_result",
    folder_tag="simple_test",
)

config = {
  "data_dir": "/home/nlp/anaconda3/envs/CogAGENT/CogAGENT/datapath/nlu/multiwoz/save_model",
  "output_dir": "/home/nlp/anaconda3/envs/CogAGENT/CogAGENT/datapath/nlu/multiwoz/output/all_context",
  "log_dir": "/home/nlp/anaconda3/envs/CogAGENT/CogAGENT/datapath/nlu/multiwoz/log/all_context",
  "DEVICE": "cuda:1",
  "seed": 2019,
  "cut_sen_len": 40,
  "use_bert_tokenizer": True,
  "model": {
    "finetune": False,
    "context": True,
    "context_grad": False,
    "pretrained_weights": "bert-base-uncased",
    "check_step": 1000,
    "max_step": 30000,
    "batch_size": 100,
    "learning_rate": 1e-5,
    "adam_epsilon": 1e-8,
    "warmup_steps": 0,
    "weight_decay": 0.0,
    "dropout": 0.1,
    "hidden_units": 1536
  }
}
raw_data_path="/home/nlp/anaconda3/envs/CogAGENT/CogAGENT/datapath/nlu/multiwoz/raw_data"
reader = JointbertReader(raw_data_path)
train_data, dev_data, test_data = reader.read_all()

intent_vocab = json.load(open(os.path.join(raw_data_path, 'intent_vocab.json')))
tag_vocab = json.load(open(os.path.join(raw_data_path, 'tag_vocab.json')))
# print('intent num:', len(intent_vocab))
# print('tag num:', len(tag_vocab))
plm = PlmAutoModel(pretrained_model_name="bert-base-uncased")
processor = MultiwozProcessor(intent_vocab=intent_vocab, tag_vocab=tag_vocab,plm="bert-base-uncased")
train_dataset = processor.process(train_data, data_key='train', cut_sen_len=config['cut_sen_len'], use_bert_tokenizer=config['use_bert_tokenizer'])
dev_dataset = processor.process(dev_data, data_key='val', cut_sen_len=config['cut_sen_len'], use_bert_tokenizer=config['use_bert_tokenizer'])
# test_dataset = processor.process(test_data, data_key='test', cut_sen_len=config['cut_sen_len'], use_bert_tokenizer=config['use_bert_tokenizer'])

model = JointbertModel(config['model'], config['DEVICE'], processor, processor.tag_dim, processor.intent_dim, processor.intent_weight)

metric = BaseJointbertMetric(mode="binary")

bcewithlogitsloss = nn.BCEWithLogitsLoss()
crossentropyloss = nn.CrossEntropyLoss()
loss = {"bcewithlogitsloss" : bcewithlogitsloss,
        "crossentropyloss" : crossentropyloss}

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=config['model']['learning_rate'])

trainer = Trainer(model,
                  train_dataset,
                #   train_data,
                  dev_data=dev_dataset,
                  n_epochs=20,
                  batch_size=100,
                  loss=loss,
                  optimizer=optimizer,
                  scheduler=None,
                  metrics=metric,
                  train_sampler=None,
                  dev_sampler=None,
                  drop_last=False,
                  gradient_accumulation_steps=1,
                  num_workers=5,
                  print_every=None,
                  scheduler_steps=None,
                  validate_steps=100,
                  save_steps=None,
                  output_path=output_path,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  fp16=False,
                  fp16_opt_level='O1',
                  collate_fn=train_dataset.to_dict,
                  dev_collate_fn=dev_dataset.to_dict,
                  )
trainer.train()
print("**************************end")
