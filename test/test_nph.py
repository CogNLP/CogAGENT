import torch.optim

from cogagent.data.readers.open_dialkg_reader import OpenDialKGReader
from cogagent.data.processors.open_dialkg_processors.open_dialkg_for_nph_processor import OpenDialKGForNPHProcessor
from cogagent import init_cogagent,Trainer,BaseClassificationMetric
from cogagent.models.neural_path_hunter_model import MaskRefineModel
from cogagent.core.metric.nph_metric import BaseNPHMetric

device, output_path = init_cogagent(
    device_id=2,
    output_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/OpenDialKG/experimental_result",
    folder_tag="run_DialoGPT",
)

reader = OpenDialKGReader(
    raw_data_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/OpenDialKG/raw_data", debug=False)
train_data,dev_data,test_data = reader.read_all()
vocab = reader.read_vocab()

plm_name = 'microsoft/DialoGPT-medium'
mlm_name = 'roberta-large'

# from cogagent import save_pickle
# save_pickle(vocab,"/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/OpenDialKG/raw_data/toolkit/vocab.pkl")
# print("Saving Finished!")

processor = OpenDialKGForNPHProcessor(vocab=vocab, plm=plm_name, mlm=mlm_name, debug=False)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
# test_dataset = processor.process_test(test_data)

# test model construction
model = MaskRefineModel(plm_name=plm_name, mlm_name=mlm_name, vocab=vocab)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=6.25e-5,eps=1e-8)
metric = BaseNPHMetric()

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=10,
                  batch_size=2,
                  loss=None,
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
                  validate_steps=1000,
                  save_by_metric="lm_ppl",
                  metric_mode='min',
                  save_steps=None,
                  output_path=output_path,
                  grad_norm=1,
                  use_tqdm=True,
                  device=device,
                  fp16=False,
                  fp16_opt_level='O1',
                  collate_fn=processor._collate,
                  )
trainer.train()
print("end")
