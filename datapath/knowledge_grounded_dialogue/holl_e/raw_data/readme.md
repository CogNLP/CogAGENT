# Holl-E Dataset
The official github repo is [here](https://github.com/nikitacs16/Holl-E).

Download the holl-e dataset from [google drive](https://drive.google.com/open?id=1xQBRDs5q_2xLOdOpbq7UeAmUM0Ht370A)
and put them into folder ```original_data```.
 
After downloading the dataset, the directory is organized like the following form.
```angular2html
datapath
├─ knowledge_grounded_dialogue
│  ├─ holl_e
│  │  ├─ raw_data
│  │  ├─ original_data
│  │  │  │  ├─ experiment_data
│  │  │  │  │  ├─ multi_reference_test.json
│  │  │  │  ├─ main_data
│  │  │  │  │  ├─ train_data.json
│  │  │  │  │  ├─ dev_data.json
│  │  │  │  │  ├─ test_data.json
│  │  │  │  ├─ raw_data
│  │  │  │  │  ├─ train_data.json
│  │  │  │  │  ├─ dev_data.json
│  │  │  │  │  ├─ test_data.json
```
Runing the preprocess code provided by [this file](./preprocess_holl_e_dataset.py)(**remember to 
modify the root path in the file!**) and the final form is listed below:

```angular2html
datapath
├─ knowledge_grounded_dialogue
│  ├─ holl_e
│  │  ├─ raw_data
│  │  │  ├─ train_data.json
│  │  │  ├─ dev_data.json
│  │  │  ├─ test_data.json
│  │  ├─ original_data
│  │  │  │  ├─ experiment_data
│  │  │  │  │  ├─ multi_reference_test.json
│  │  │  │  ├─ main_data
│  │  │  │  │  ├─ train_data.json
│  │  │  │  │  ├─ dev_data.json
│  │  │  │  │  ├─ test_data.json
│  │  │  │  ├─ raw_data
│  │  │  │  │  ├─ train_data.json
│  │  │  │  │  ├─ dev_data.json
│  │  │  │  │  ├─ test_data.json
```