import torch.nn as nn
import torch.optim as optim
from cogagent import init_cognlp, ModReader, ModProcessor, StickerDialogModel, \
    BaseTopKMetric, Trainer

device, output_path = init_cognlp(
    device_id=2,
    output_path="/data/mentianyi/code/CogAGENT/datapath/mm_dialog/mod/experimental_result",
    folder_tag="final",
)

reader = ModReader(raw_data_path="/data/mentianyi/code/CogAGENT/datapath/mm_dialog/mod/raw_data")
train_data, dev_data, test_data = reader.read_all()
addition = reader.read_addition()

processor = ModProcessor(
    pretrained_model_name_or_path='YeungNLP/clip-vit-bert-chinese-1M',
    max_token_len=510,
    addition=addition,
    do_sample=False)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)
addition = processor.get_addition()

model = StickerDialogModel(max_image_id=307,
                           pretrained_image_model_name_or_path='YeungNLP/clip-vit-bert-chinese-1M',
                           pretrained_model_name_or_path='bert-base-chinese',
                           pretrained_image_tokenizer_name_or_path='BertTokenizerFast',
                           addition=addition,
                           image_path="/data/mentianyi/code/CogAGENT/datapath/mm_dialog/mod/raw_data/meme_set")
metric = BaseTopKMetric(topk=[1, 2, 5])
loss = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),
                        lr=0.000001,
                        betas=(0.9, 0.98),
                        weight_decay=0.2)

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=10000,
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
                  validate_steps=10000,
                  save_steps=10000,
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

