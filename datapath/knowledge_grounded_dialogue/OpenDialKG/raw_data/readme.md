# OpenDialKG Dataset
The official github repo is [here](https://github.com/facebookresearch/opendialkg).

Reference: https://github.com/nouhadziri/Neural-Path-Hunter

Download the dataset from [here](https://raw.githubusercontent.com/facebookresearch/opendialkg/main/data/opendialkg.csv) and put them in the following form:
```angular2html
datapath
├─ knowledge_grounded_dialogue
│  ├─ OpenDialKG
│  │  ├─ raw_data
│  │  │  ├─ original_data
│  │  │  │  ├─ opendialkg.csv
```

Runing the preprocess code provided by [this file](./preprocess_opendialkg.py) and the final form is listed below:
 
```angular2html
datapath
├─ knowledge_grounded_dialogue
│  ├─ OpenDialKG
│  │  ├─ raw_data
│  │  │  ├─ train_nph_data.json
│  │  │  ├─ dev_nph_data.json
│  │  │  ├─ test_nph_data.json
│  │  │  ├─ original_data
│  │  │  │  ├─ opendialkg.csv
```

Then, download the [graph embedding](https://drive.google.com/drive/folders/1KzjCq0-8K1pqi1TFfsEC3iKiaK-2oL1I) and then put the in the following form:

```angular2html
datapath
├─ knowledge_grounded_dialogue
│  ├─ OpenDialKG
│  │  ├─ raw_data
│  │  │  ├─ train_nph_data.json
│  │  │  ├─ dev_nph_data.json
│  │  │  ├─ test_nph_data.json
│  │  │  ├─ graph_embedding
│  │  │  │  ├─ entities.txt
│  │  │  │  ├─ relations.txt
│  │  │  │  ├─ entity_embedding.npy
│  │  │  │  ├─ relation_embedding.npy
│  │  │  ├─ original_data
│  │  │  │  ├─ opendialkg.csv
```