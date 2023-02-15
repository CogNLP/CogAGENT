import pytrec_eval
import torch
import numpy as np
import random
import json
import logging

from cogagent.core.metric.MMCoQA_Metric import MMCoQAMetric
from cogagent.data.readers.MMCoQA_reader import MMCoQAReader
from cogagent.data.processors.MMCoQA_processors.MMCoQA_processor import MMCoQAProcessor
from cogagent.models.MMCoQA.MMCoQA_model import Pipeline, BertForOrconvqaGlobal, BertForRetrieverOnlyPositivePassage
import torch.nn as nn
import torch.optim as optim


import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogagent import *
from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"

device, output_path = init_cogagent(
    device_id=6,
    output_path="/data/yangcheng/CogAgent/datapath/MMCoQA_data/experimental_result/MMCoQA_train_test/",
    folder_tag="MMCoQA_test",
)

MODEL_CLASSES = {
    'reader': (BertConfig, BertForOrconvqaGlobal, BertTokenizer),
    'retriever': (BertConfig, BertForRetrieverOnlyPositivePassage, BertTokenizer),
}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

scheduler = None

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# args.reader_model_type = args.reader_model_type.lower()
reader_cache_dir = "/data/yangcheng/huggingface_cache/bert-base-uncased/"
reader_config_class, reader_model_class, reader_tokenizer_class = MODEL_CLASSES['reader']
reader_config = reader_config_class.from_pretrained("/data/yangcheng/CogAgent/datapath/huggingface_cache/bert-base-uncased/",
                                                    cache_dir=reader_cache_dir if reader_tokenizer_class else None)
reader_config.num_qa_labels = 2
# this not used for BertForOrconvqaGlobal
reader_config.num_retrieval_labels = 2
reader_config.qa_loss_factor = 1.0
reader_config.retrieval_loss_factor = 1.0
reader_config.proj_size = 128

reader_tokenizer = reader_tokenizer_class.from_pretrained("/data/yangcheng/CogAgent/datapath/huggingface_cache/bert-base-uncased/",
                                                          do_lower_case=True,
                                                          cache_dir=reader_cache_dir if reader_cache_dir else None)
# reader_tokenizer.pad_token = reader_tokenizer.eos_token
reader_model = reader_model_class.from_pretrained("/data/yangcheng/CogAgent/datapath/huggingface_cache/bert-base-uncased/",
                                                  from_tf=bool('.ckpt' in 'bert-base-uncased'),
                                                  config=reader_config,
                                                  cache_dir=reader_cache_dir if reader_cache_dir else None)

# args.retriever_model_type = args.retriever_model_type.lower()
retriever_config_class, retriever_model_class, retriever_tokenizer_class = MODEL_CLASSES['retriever']
retriever_config = retriever_config_class.from_pretrained("/data/yangcheng/CogAgent/datapath/MMCoQA_data/checkpoint-18000/")

# load pretrained retriever
retriever_tokenizer = retriever_tokenizer_class.from_pretrained("/data/yangcheng/CogAgent/datapath/MMCoQA_data/checkpoint-18000/")
retriever_model = retriever_model_class.from_pretrained("/data/yangcheng/CogAgent/datapath/MMCoQA_data/checkpoint-18000/", force_download=True)

model = Pipeline(reader_tokenizer=reader_tokenizer)
model.retriever = retriever_model
# do not need and do not tune passage encoder
model.retriever.passage_encoder = None
model.retriever.passage_proj = None
model.retriever.image_encoder = None
model.retriever.image_proj = None
model.reader = reader_model

reader = MMCoQAReader(raw_data_path="/data/yangcheng/CogAgent/datapath/MMCoQA_data/data/")
train_data, dev_data, test_data = reader.read_all()
batch = reader.read_addition()

processor = MMCoQAProcessor()
train_dataset = processor.process_train(train_data['train_data'], batch)
dev_dataset = processor.process_dev(dev_data['dev_data'], batch)
# test_dataset = processor.process_test(test_data['test_data'], batch, device)

qrels = batch['qrels']
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg', 'set_recall'})
metric = MMCoQAMetric(evaluator=evaluator)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
trainer = Trainer(model,  # 模型
                  train_dataset,  # 训练数据
                  dev_data=dev_dataset,  # 验证数据/
                  n_epochs=10,
                  batch_size=2,
                  loss=loss,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  metrics=metric,
                  train_sampler=None,
                  dev_sampler=None,
                  drop_last=True,
                  gradient_accumulation_steps=1,
                  num_workers=5,
                  print_every=None,
                  scheduler_steps=None,
                  validate_steps=3000,  # 每多少步验证一下指标
                  save_steps=3000,  # 多少步存一下
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
