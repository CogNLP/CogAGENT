import torch.nn as nn
import torch.optim as optim
from cogagent import *
from cogagent.core.evaluator import Evaluator

device, output_path = init_cogagent(
    device_id=2,
    output_path="/data/hongbang/CogAGENT/datapath/text_classification/SST_2/experimental_result",
    folder_tag="simple_test",
)

reader = Sst2Reader(raw_data_path="/data/hongbang/CogAGENT/datapath/text_classification/SST_2/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

processor = Sst2Processor(plm="bert-base-cased", max_token_len=128, vocab=vocab)
# train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
# test_dataset = processor.process_test(test_data)

plm = PlmAutoModel(pretrained_model_name="bert-base-cased")
model = BaseTextClassificationModel(plm=plm, vocab=vocab)
metric = BaseClassificationMetric(mode="binary")
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

evaluator = Evaluator(
    model=model,
    checkpoint_path="/data/hongbang/CogAGENT/datapath/text_classification/SST_2/experimental_result/simple_test--2022-11-03--06-42-31.77/best_model/checkpoint-500",
    dev_data=dev_dataset,
    metrics=metric,
    sampler=None,
    drop_last=False,
    file_name="models.pt",
    batch_size=32,
    device=device,
    user_tqdm=True,
)
evaluator.evaluate()
print("end")
