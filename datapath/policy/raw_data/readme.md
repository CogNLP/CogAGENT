# The MultiWOZ Dataset
Download  MultiWOZ dataset .    

Before training or testing, you need to download training data and database files.

data_path: https://convlab.blob.core.windows.net/convlab-2/mdrg_data.zip,

db_path: https://convlab.blob.core.windows.net/convlab-2/mdrg_db.zip

Put the raw data in path`datapath/policy/raw_data`.   
Like the following form:

```angular2html
datapath
├─ policy
│  ├─ raw_data
│  │  ├─ db
│  │  ├─ train_dials.json
│  │  ├─ val_dials.json
│  │  ├─ test_dials.json
│  │  ├─ delex.json
│  │  ├─ input_lang.index2word.json
│  │  ├─ input_lang.word2index.json
│  │  ├─ output_lang.index2word.json
│  │  └─ output_lang.word2index.json
```