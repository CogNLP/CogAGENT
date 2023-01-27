import torch.nn as nn
import torch.optim as optim
from cogagent import init_cognlp, MMchatReader, MMchatProcessor, MMchatDialog, \
    BaseDialogMetric, Trainer

device, output_path = init_cognlp(
    device_id=1,
    output_path="/data/mentianyi/code/CogAGENT/datapath/mm_dialog/mmchat/experimental_result",
    folder_tag="simple_test",
)

reader = MMchatReader(
    raw_data_path="/data/mentianyi/code/CogAGENT/datapath/mm_dialog/mmchat/raw_data")
train_data, dev_data, test_data = reader.read_all()

processor = MMchatProcessor(
    pretrained_model_name_or_path="uer/gpt2-chinese-cluecorpussmall",
    max_token_len=512,
    valid_mod="serial_valid",
    do_sample=False)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(train_data)
test_dataset = processor.process_test(train_data)
addition = processor.get_addition()

model = MMchatDialog(
    generate_max_len=20,
    addition=addition,
    valid_mod="serial_valid",
    pretrained_model_name_or_path="uer/gpt2-chinese-cluecorpussmall")
metric = BaseDialogMetric(valid_mod="serial_valid")
loss = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.00001)

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=1000,
                  batch_size=5,
                  loss=loss,
                  optimizer=optimizer,
                  scheduler=None,
                  metrics=metric,
                  dev_sampler=None,
                  drop_last=False,
                  gradient_accumulation_steps=1,
                  num_workers=5,
                  print_every=None,
                  scheduler_steps=None,
                  validate_steps=1000,
                  save_steps=42315,
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
