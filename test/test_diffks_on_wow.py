from cogagent.data.processors.holl_e_processors.holl_e_for_diffks_processor import HollEForDiffksProcessor
from cogagent.models.diffks_model import DiffKSModel
from cogagent.core.metric.base_kgc_metric import BaseKGCMetric
from cogagent import *
import torch
import torch.nn as nn
import torch.optim as optim

device, output_path = init_cogagent(
    device_id=8,
    output_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/experimental_result",
    folder_tag="run_diffks_wow_lr1e-4",
)

# cache_file = "/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/cache/processor_datas_debug.pkl"
# train_dataset, dev_dataset, test_dataset,vocab = load_pickle(cache_file)
# train_dataset = torch.utils.data.Subset(train_dataset,list(range(4*98,len(train_dataset))))
# dev_dataset = torch.utils.data.Subset(dev_dataset,list(range(4*105,len(dev_dataset))))

reader = WoWReader(raw_data_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

processor = WoWForDiffksProcessor(max_token_len=512, vocab=vocab, debug=False)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)

model = DiffKSModel(glove_path='/data/hongbang/CogAGENT/datapath/pretrained_models/glove', vocab=vocab)
metric = BaseKGCMetric(default_metric_name="bleu-4",vocab=vocab)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 新加了注释
trainer = Trainer(model,
                  train_dataset,
                  dev_data=test_dataset,
                  n_epochs=40,
                  batch_size=2,
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
                  validate_steps=2000,
                  save_by_metric="bleu-4",
                  save_steps=None,
                  output_path=output_path,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  fp16=False,
                  fp16_opt_level='O1',
                  )
trainer.train()
print("end")


