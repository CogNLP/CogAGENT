---


# <img src="docs/images/log.png" alt="*CogAGENT: A Multimodal, Knowledgeable and Controllable Toolkit for Building Conversational Agents" width="50%">
**CogAGENT: A Multimodal, Knowledgeable and Controllable Toolkit for Building Conversational Agents**

**Demo system and more information is available at https://synehe.github.io/CogAGENT/**
**A short illustration video is at https://youtu.be/SE0SEeiAmXI**



## Description
CogAGENT is a toolkit for building **multimodal**, **knowledgeable** and **controllable** conversational agents. We provide 17 models and integrate a variety of datasets covered above features. We decouple and modularize them flexibly to make users more convenient for development and research.

This package has the following advantages:

- **A multimodal, knowledgeable and controllable conversational framework.** We propose a unified framework named CogAGENT, incorporating Multimodal Module, Knowledgeable Module and Controllable Module to conduct multimodal interaction, generate knowledgeable response and make replies under control in real scenarios.
- **Comprehensive conversational models, datasets and metrics.** CogAGENT implements 17 conversational models covering task-oriented dialogue, open-domain dialogue and question-answering tasks.
  We also integrate some widely used conversational datasets and metrics to verify the performance of models.
- **Open-source and modularized conversational toolkit.** We release CogAGENT as an open-source toolkit and modularize conversational agents to provide easy-to-use interfaces. Hence, users can modify codes for their own customized models or datasets.
- **Online dialogue system.** We release an online system, which supports conversational agents to interact with users. We also provide a  video to illustrate how to use it.

## Install

### Install from git

```bash
# clone CogAGENT 
git git@github.com:CogNLP/CogAGENT.git

# install CogAGENT  
cd cogagent
pip install -e .   
pip install -r requirements.txt
```

## Quick Start

### Programming Framework for Training Models

```python
from cogagent import *
import torch
import torch.nn as nn
import torch.optim as optim

# init the logger,device and experiment result saving dir
device, output_path = init_cogagent(
    device_id=8,
    output_path=datapath,
    folder_tag="run_diffks_on_wow",
)

# choose utterance reader
reader = WoWReader(raw_data_path=raw_data_path)
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

# choose data processor 
# In the training phase, no retriever is selected as the knowledge is provided by dataset
processor = WoWForDiffksProcessor(max_token_len=512, vocab=vocab, debug=False)
train_dataset = processor.process_train(train_data)
dev_dataset = processor.process_dev(dev_data)
test_dataset = processor.process_test(test_data)

# choose response generator
model = DiffKSModel()
metric = BaseKGCMetric(default_metric_name="bleu-4",vocab=vocab)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Use the provided Trainer class to start the model training process
trainer = Trainer(model,train_dataset,dev_data=test_dataset,n_epochs=40,batch_size=2,
                  loss=loss,optimizer=optimizer,scheduler=None,metrics=metric,
                  drop_last=False,gradient_accumulation_steps=1,num_workers=5,
                  validate_steps=2000,save_by_metric="bleu-4",save_steps=None,
                  output_path=output_path,grad_norm=1,
                  use_tqdm=True,device=device,
                  fp16_opt_level='O1',
                  )
trainer.train()
```


 