import torch.nn as nn
import torch.optim as optim
from cogagent import *
from cogagent.core.evaluator import Evaluator
from cogagent.models.diffks_model import DiffKSModel
from cogagent.core.metric.base_kgc_metric import BaseKGCMetric

device, output_path = init_cogagent(
    device_id=8,
    output_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/experimental_result",
    folder_tag="debug_inference_diffks_wow",
)

reader = WoWReader(raw_data_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

processor = WoWForDiffksProcessor(max_token_len=512, vocab=vocab, debug=False)
# train_dataset = processor.process_train(train_data)
# dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)

model = DiffKSModel(glove_path='/data/hongbang/CogAGENT/datapath/pretrained_models/glove', vocab=vocab)
metric = BaseKGCMetric(default_metric_name="bleu-4",vocab=vocab)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

evaluator = Evaluator(
    model=model,
    checkpoint_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/experimental_result/run_diffks_wow--2022-11-13--04-17-09.20/best_model/checkpoint-9215",
    dev_data=test_dataset,
    metrics=metric,
    sampler=None,
    drop_last=False,
    file_name="models.pt",
    batch_size=2,
    device=device,
    user_tqdm=True,
)
evaluator.evaluate()
print("end")
