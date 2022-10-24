from cogagent.data.processors.holl_e_processors.holl_e_for_diffks_processor import HollEForDiffksProcessor
from cogagent.models.diffks_model import DiffKSModel
from cogagent import *
import torch
import torch.nn as nn
import torch.optim as optim

device, output_path = init_cogagent(
    device_id=2,
    output_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/holl_e/experimental_result",
    folder_tag="debug_diffks",
)

cache_file = "/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/holl_e/cache/processor_datas.pkl"
train_dataset, dev_dataset, test_dataset,vocab = load_pickle(cache_file)

model = DiffKSModel(glove_path='/data/hongbang/CogAGENT/datapath/pretrained_models/glove', vocab=vocab)
metric = BaseClassificationMetric(mode="binary")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# 新加了注释
trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=20,
                  batch_size=8,
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
                  save_by_metric="F1",
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


