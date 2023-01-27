import torch.nn as nn
import torch.optim as optim
from cogagent import init_cognlp, ChineseGossipDialogReader, ChineseGossipDialogProcessor, ChineseGossipDialog, \
    BaseDialogMetric, Trainer

device, output_path = init_cognlp(
    device_id=2,
    output_path="/data/mentianyi/code/CogAGENT/datapath/gossip_dialog/chinese_gossip_dialog/experimental_result",
    folder_tag="final",
)

reader = ChineseGossipDialogReader(
    raw_data_path="/data/mentianyi/code/CogAGENT/datapath/gossip_dialog/chinese_gossip_dialog/raw_data")
train_data, dev_data, test_data = reader.read_all()

processor = ChineseGossipDialogProcessor(
    pretrained_model_name_or_path="uer/gpt2-chinese-cluecorpussmall",
    max_token_len=512,
    valid_mod="serial_valid",
    do_sample=False)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(dev_data)
addition = processor.get_addition()

model = ChineseGossipDialog(
    pretrained_model_name_or_path="uer/gpt2-chinese-cluecorpussmall",
    generate_max_len=20,
    addition=addition,
    valid_mod="serial_valid")
metric = BaseDialogMetric(valid_mod="serial_valid")
loss = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.00001)

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=1000,
                  batch_size=20,
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
                  validate_steps=5000,
                  save_steps=5000,
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
print("end")
