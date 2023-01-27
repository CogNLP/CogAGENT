import torch.nn as nn
import torch.optim as optim
from cogagent import init_cognlp, WayReader, WayProcessor, LedDialog, \
    BaseEnbodiedMetric, Trainer

device, output_path = init_cognlp(
    device_id=0,
    output_path="/data/mentianyi/code/CogAGENT/datapath/embodied_dialog/way/experimental_result",
    folder_tag="final",
)

reader = WayReader(raw_data_path="/data/mentianyi/code/CogAGENT/datapath/embodied_dialog/way/raw_data")
train_data, dev_seen_data, dev_unseen_data, test_data = reader.read_all()
addition = reader.read_addition()

processor = WayProcessor(
    addition=addition,
    do_sample=False)
train_dataset = processor.process_train(train_data)
dev_seen_dataset = processor.process_dev_seen(dev_seen_data)
dev_unseen_dataset = processor.process_dev_unseen(dev_unseen_data)
# test_dataset = processor.process_test(test_data)
addition = processor.get_addition()

model = LedDialog(addition=addition,
                  embedding_dir="/data/mentianyi/code/CogAGENT/datapath/embodied_dialog/way/raw_data/word_embeddings/")
metric = BaseEnbodiedMetric(default_metric_name="acc0m")
loss = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(model.parameters(), lr=0.00001)

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_seen_dataset,
                  n_epochs=1000,
                  batch_size=30,
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
                  validate_steps=300,
                  save_steps=300,
                  output_path=output_path,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  fp16=False,
                  fp16_opt_level='O1',
                  collate_fn=train_dataset.to_dict,
                  dev_collate_fn=dev_seen_dataset.to_dict,
                  )
trainer.train()
print("end")
