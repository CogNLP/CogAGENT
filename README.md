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

## AVAILABEL MODELS OF COGAGENT

<table class="greyGridTable" style="border:1px solid black;" >
  <!--					<table class="w3-hover-green" >-->
  <thead>
  <tr style="border:1px solid black;">
      <th style="border:1px solid black;">Modal</th>
      <th style="border:1px solid black;">Category</th>
      <th style="border:1px solid black;">Reference</th>
  </tr>
  </thead>
  <tbody>

  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          SUMBT
      </td>
      <td style="border:1px solid black;">Fundamental</td>
      <td style="border:1px solid black;">SUMBT: Slot-Utterance Matching for Universal and Scalable Belief Tracking</td>
  </tr>

  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          SC-LSTM
      </td>
      <td style="border:1px solid black;">Fundamental</td>
      <td style="border:1px solid black;">Semantically conditioned lstm-based natural language generation for spoken dialogue systems</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          BERTNLU
      </td>
      <td style="border:1px solid black;">Fundamental</td>
      <td style="border:1px solid black;">ConvLab-2: An Open-Source Toolkit for Building, Evaluating, and Diagnosing Dialogue Systems</td>
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          MDRG
      </td>
      <td style="border:1px solid black;">Fundamental</td>
      <td style="border:1px solid black;">Towards end-to-end multi-domain dialogue modelling</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          UBAR
      </td>
      <td style="border:1px solid black;">Fundamental</td>
      <td style="border:1px solid black;">owards fully end-to-end task-oriented dialog system with gpt-</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          GPT2 for Chinese chitchat
      </td>
      <td style="border:1px solid black;">Fundamental</td>
      <td style="border:1px solid black;">Chinese chitchat</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          TransResNet-Ret
      </td>
      <td style="border:1px solid black;">Multimodal</td>
      <td style="border:1px solid black;">Image-Chat: Engaging Grounded Conversations</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          MMBERT
      </td>
      <td style="border:1px solid black;">Multimodal</td>
      <td style="border:1px solid black;">Selecting Stickers in Open-Domain Dialogue through Multitask Learning</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          MAE
      </td>
      <td style="border:1px solid black;">Multimodal</td>
      <td style="border:1px solid black;">MMCoQA: Conversational Question Answering over Text, Tables, and Images</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          PICa
      </td>
      <td style="border:1px solid black;">Multimodal</td>
      <td style="border:1px solid black;">An empirical study of gpt-3 for few-shot knowledge-based vqa</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          LingUNet 
      </td>
      <td style="border:1px solid black;">Multimodal</td>
      <td style="border:1px solid black;">Where Are You? Localization from Embodied Dialog</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          DifffKS
      </td>
      <td style="border:1px solid black;">Knowledgeable</td>
      <td style="border:1px solid black;">Difference-aware Knowledge Selection for Knowledge-grounded Conversation Generation</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          KE-Blender
      </td>
      <td style="border:1px solid black;">Knowledgeable</td>
      <td style="border:1px solid black;">Knowledge Enhanced Fine-Tuning for Better Handling Unseen Entities in Dialogue Generation</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          NPH
      </td>
      <td style="border:1px solid black;">Knowledgeable</td>
      <td style="border:1px solid black;">Neural Path Hunter: Reducing Hallucination in Dialogue Systems via Path Grounding</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          BERTQA
      </td>
      <td style="border:1px solid black;">Knowledgeable</td>
      <td style="border:1px solid black;">Dense Passage Retrieval for Open-Domain Question Answering</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          KEMP
      </td>
      <td style="border:1px solid black;">Controllable</td>
      <td style="border:1px solid black;">OpenDialKG: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          RobertaClassifier
      </td>
      <td style="border:1px solid black;">Controllable</td>
      <td style="border:1px solid black;">On the Safety of Conversational Models: Taxonomy, Dataset, and Benchmark</td>
      
  </tr>

  </tbody>
</table>



## AVAILABEL DATASETS OF COGAGENT


<table class="greyGridTable" style="border:1px solid black;" >
  <!--					<table class="w3-hover-green" >-->
  <thead>
  <tr style="border:1px solid black;">
      <th style="border:1px solid black;">Dataset</th>
      <th style="border:1px solid black;">Category</th>
      <th style="border:1px solid black;">Reference</th>
  </tr>
  </thead>
  <tbody>

  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          MultiWOZ 2.0
      </td>
      <td style="border:1px solid black;">Fundamental</td>
      <td style="border:1px solid black;">MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling</td>
  </tr>

  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          MultiWOZ 2.1
      </td>
      <td style="border:1px solid black;">Fundamental</td>
      <td style="border:1px solid black;">MultiWOZ 2.1: A Consolidated Multi-Domain Dialogue Dataset with State Corrections and State Tracking Baselines</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          Chinese chitchat Dataset
      </td>
      <td style="border:1px solid black;">Fundamental</td>
      <td style="border:1px solid black;">Chinese chitchat</td>
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          MOD
      </td>
      <td style="border:1px solid black;">Multimodal</td>
      <td style="border:1px solid black;">DSTC10-Track1</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          MMConvQA
      </td>
      <td style="border:1px solid black;">Multimodal</td>
      <td style="border:1px solid black;">MMCoQA: Conversational Question Answering over Text, Tables, and Images</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          OK-VQA
      </td>
      <td style="border:1px solid black;">Multimodal</td>
      <td style="border:1px solid black;">Ok-vqa: A visual question answering benchmark requiring external knowledge</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          VQAv2
      </td>
      <td style="border:1px solid black;">Multimodal</td>
      <td style="border:1px solid black;">Making the v in vqa matter: Elevating the role of image understanding in visual question answering</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          WAY
      </td>
      <td style="border:1px solid black;">Multimodal</td>
      <td style="border:1px solid black;">Where Are You? Localization from Embodied Dialog</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          Wizard of Wikipedia
      </td>
      <td style="border:1px solid black;">Knowledgeable</td>
      <td style="border:1px solid black;">Wizard of Wikipedia: Knowledge-Powered Conversational Agents</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          Holl-E
      </td>
      <td style="border:1px solid black;">Knowledgeable</td>
      <td style="border:1px solid black;">Towards Exploiting Background Knowledge for Building Conversation Systems</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          OpenDialKG 
      </td>
      <td style="border:1px solid black;">Knowledgeable</td>
      <td style="border:1px solid black;">OpenDialKG: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          DIASAFETY
      </td>
      <td style="border:1px solid black;">Controllable</td>
      <td style="border:1px solid black;">On the Safety of Conversational Models: Taxonomy, Dataset, and Benchmark</td>
      
  </tr>
  <tr style="border:1px solid black;">
      <td style="border:1px solid black;">
          EmpatheticDialogues
      </td>
      <td style="border:1px solid black;">Controllable</td>
      <td style="border:1px solid black;">Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset</td>
      
  </tr>

  </tbody>
</table>

