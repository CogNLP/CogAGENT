# MMCoQA

## Datasets
The MMCoQA dataset can be downloaded via this link, "https://github.com/liyongqi67/MMCoQA".
## reader
Use MMCoQA_reader.py to read the train data,valid data, test data and retrieval dictionaries.
Pass the dataset path to MMCoQAReader() to read the data.
Such as:
```
reader = MMCoQAReader(raw_data_path="/data/yangcheng/CogAgent/datapath/MMCoQA_data/data/")
```

## processors
After reading the datas into memory,use the MMCoQA_processors.py to process the data.

## train
Use the MMCoQA_train.py to train the model.

reader_model,retriever_model can be set according to your preference.

As long as you set the local path, you can load both the transformers model and the local model.
Such as:
```
retriever_model = retriever_model_class.from_pretrained("/data/yangcheng/CogAgent/datapath/MMCoQA_data/checkpoint-18000/")
```

