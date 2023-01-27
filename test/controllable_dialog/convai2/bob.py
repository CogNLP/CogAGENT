import torch.nn as nn
import torch.optim as optim
from cogagent import init_cognlp, Convai2Reader, Convai2Processor, Bob, \
    BasePerplexityMetric, Trainer

device, output_path = init_cognlp(
    device_id=4,
    output_path="/data/mentianyi/code/CogAGENT/datapath/controllable_dialog/convai2/experimental_result",
    folder_tag="final",
)

reader = Convai2Reader(raw_data_path="/data/mentianyi/code/CogAGENT/datapath/controllable_dialog/convai2/raw_data")
train_data, dev_data, test_data = reader.read_all()
addition = reader.read_addition()

processor = Convai2Processor(
    pretrained_model_name_or_path='bert-base-uncased',
    max_source_len=64,
    max_target_len=32,
    do_sample=False,
    addition=addition)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)
addition = processor.get_addition()

model = Bob(
    encoder_model='bert-base-uncased',
    decoder_model='bert-base-uncased',
    decoder_model2='bert-base-uncased',
    nli_batch_size=32,
    addition=addition)
metric = BasePerplexityMetric(default_metric_name="sents_ppl_1",
                              display_example=[1])
loss = None
optimizer = optim.Adam(model.parameters(), lr=0.00001)

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=1000,
                  batch_size=80,
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
                  validate_steps=800,
                  save_steps=800,
                  output_path=output_path,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  fp16=False,
                  fp16_opt_level='O1',
                  collate_fn=train_dataset.to_dict,
                  dev_collate_fn=dev_dataset.to_dict,
                  metric_mode="min"
                  )
trainer.train()
print("end")
