import torch.nn as nn
import torch.optim as optim
from cogagent import *
from cogagent.data.readers.diasafety_reader import DiaSafetyReader
from cogagent.data.processors.dia_safety_processors.dia_safety_for_classify_processor import DiaSafetyForClassifyProcessor

# ['agreement', 'expertise', 'offend','bias','risk']
category = "risk"

device, output_path = init_cogagent(
    device_id=2,
    output_path="/data/hongbang/CogAGENT/datapath/dialogue_safety/DiaSafety/experiment",
    folder_tag="train_"+category,
)

reader = DiaSafetyReader(raw_data_path="/data/hongbang/CogAGENT/datapath/dialogue_safety/DiaSafety/raw_data",category=category)
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

plm = 'roberta-base'
processor = DiaSafetyForClassifyProcessor(plm=plm, max_token_len=128, vocab=vocab, debug=False)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)

plm = PlmAutoModel(pretrained_model_name=plm)
model = BaseTextClassificationModel(plm=plm, vocab=vocab)
metric = BaseClassificationMetric(mode="multi")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# 新加了注释
trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=10,
                  batch_size=32,
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
                  save_by_metric="macro_F1",
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
