# Wizards of Wikipedia Dataset
The official github repo is [here](https://parl.ai/projects/wizard_of_wikipedia/).

Download the holl-e dataset from [one drive](https://1drv.ms/f/s!Aky8v8NZbQx1qj8xxqkdpzcSdTkd)
and put them into folder ```original```.
 
After downloading the dataset, the directory is organized like the following form.
```angular2html
datapath
├─ knowledge_grounded_dialogue
│  ├─ wow
│  │  ├─ raw_data
│  │  ├─ original 
│  │  │  │  ├─ train_collected.json
│  │  │  │  ├─ valid_collected.json
│  │  │  │  ├─ valid_unseen_collected.json
│  │  │  │  ├─ test_collected.json
│  │  │  │  ├─ test_unseen_collected.json
```
Runing the preprocess code provided by [this file](./preprocess_wow.py) and the final form is listed below:

```angular2html
datapath
├─ knowledge_grounded_dialogue
│  ├─ holl_e
│  │  ├─ raw_data
│  │  │  ├─ train.json
│  │  │  ├─ dev.json
│  │  │  ├─ valid_random_split.json
│  │  │  ├─ valid_topic_split.json
│  │  │  ├─ test_random_split.json
│  │  │  ├─ test_topic_split.json
│  │  │  ├─ test_seen.json
│  │  │  ├─ test_unseen.json
│  │  ├─ original 
│  │  │  │  ├─ train_collected.json
│  │  │  │  ├─ valid_collected.json
│  │  │  │  ├─ valid_unseen_collected.json
│  │  │  │  ├─ test_collected.json
│  │  │  │  ├─ test_unseen_collected.json
```