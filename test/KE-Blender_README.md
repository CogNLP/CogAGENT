# KE-Blender

## Datasets
The dataset uses wizard_of_wikipedia.

The preprocessed training dataset can be downloaded from...

The datasets should contain these eight files, 
including data.json,test_random_spilt.json,test_topic_split.json, topic_splits.json,train.json,train_sim.text,valid_random_split.json,valid_topic_split.json. 

## reader
Use KE_Blender_reader.py to split and read 'train.json','valid_topic_split.json','test_topic_split.json' ,'train_sim.txt' to be the train data,valid data and test data.
Pass the dataset path to KE_BlenderReader() to read the data.
Such as:
```
reader = KE_BlenderReader(raw_data_path="/home/nlp/CogAGENT/datapath/KE_Blender_data/")
```

## processors
After reading the datas into memory,use the KE_Blender_processors.py to process the data.

## train
Use the KE_Blender_base.py to train the model.

Encoder_tokenizer,decoder_tokenizer and model can be set according to your preference.

As long as you set the local path, you can load both the transformers model and the local model.
Such as:
```
model = BlenderbotSmallForConditionalGeneration.from_pretrained("/home/nlp/CogAGENT/datapath/KE_Blender_data")
```

