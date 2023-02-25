import streamlit as st
from PIL import Image


st.sidebar.markdown("# Main page 🎈")

image = Image.open('../../docs/log.png')
st.image(image)

st.title('A :blue[Multimodal], :green[Knowledgeable] and :orange[Controllable] Toolkit for Building Conversational Agents :robot_face: ')


st.header('Feature')

st.subheader('**1.A multimodal, knowledgeable and controllable conversational framework.**')
st.markdown('We propose a unified framework named CogAGENT, incorporating Multimodal Module, Knowledgeable Module and Controllable Module to conduct multimodal interaction, generate knowledgeable response and make replies under control in real scenarios.</font>')
st.subheader('**2.Comprehensive conversational models, datasets and metrics.**')
st.markdown('CogAGENT implements 17 conversational models covering task-oriented dialogue, open-domain dialogue and question-answering tasks. We also integrate some widely used conversational datasets and metrics to verify the performance of models.')
st.subheader('**3.Open-source and modularized conversational toolkit.**')
st.markdown('We release CogAGENT as an open-source toolkit and modularize conversational agents to provide easy-to-use interfaces. Hence, users can modify codes for their own customized models or datasets.')
st.subheader('**4.Online dialogue system.**')
st.markdown('We release an online system, which supports conversational agents to interact with users. We also provide a  video to illustrate how to use it.')

# st.header('2.Demo Usage')


# st.header('3.Code Usage')
#
# st.subheader('3.1 Quick Start')
# code = '''
# # clone CogAGENT
# git git@github.com:CogNLP/CogAGENT.git
#
# # install CogAGENT
# cd cogagent
# pip install -e .
# pip install -r requirements.txt
# '''
# st.code(code, language='bash')
#
# st.subheader('3.2 Programming Framework for Training Models')
#
# code = '''
# from cogagent import *
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# # init the logger,device and experiment result saving dir
# device, output_path = init_cogagent(
#     device_id=8,
#     output_path=datapath,
#     folder_tag="run_diffks_on_wow",
# )
#
# # choose utterance reader
# reader = WoWReader(raw_data_path=raw_data_path)
# train_data, dev_data, test_data = reader.read_all()
# vocab = reader.read_vocab()
#
# # choose data processor
# # In the training phase, no retriever is selected as the knowledge is provided by dataset
# processor = WoWForDiffksProcessor(max_token_len=512, vocab=vocab, debug=False)
# train_dataset = processor.process_train(train_data)
# dev_dataset = processor.process_dev(dev_data)
# test_dataset = processor.process_test(test_data)
#
# # choose response generator
# model = DiffKSModel()
# metric = BaseKGCMetric(default_metric_name="bleu-4",vocab=vocab)
# loss = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
#
# # Use the provided Trainer class to start the model training process
# trainer = Trainer(model,train_dataset,dev_data=test_dataset,n_epochs=40,batch_size=2,
#                   loss=loss,optimizer=optimizer,scheduler=None,metrics=metric,
#                   drop_last=False,gradient_accumulation_steps=1,num_workers=5,
#                   validate_steps=2000,save_by_metric="bleu-4",save_steps=None,
#                   output_path=output_path,grad_norm=1,
#                   use_tqdm=True,device=device,
#                   fp16_opt_level='O1',
#                   )
# trainer.train()
#
# '''
#
#
# st.code(code, language='python')


# import pandas as pd
# import streamlit as st
#
# # Cache the dataframe so it's only loaded once
# @st.cache_data
# def load_data():
#     return pd.DataFrame(
#         {
#             "first column": [1, 2, 3, 4],
#             "second column": [10, 20, 30, 40],
#         }
#     )
#
# # Boolean to resize the dataframe, stored as a session state variable
# st.checkbox("Use container width", value=False, key="use_container_width")
#
# df = load_data()
#
# # Display the dataframe and allow the user to stretch the dataframe
# # across the full width of the container, based on the checkbox value
# st.dataframe(df, use_container_width=st.session_state.use_container_width)

