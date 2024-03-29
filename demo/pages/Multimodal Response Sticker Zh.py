import sys

sys.path.append('/data/mentianyi/code/CogNLP')
import streamlit as st
from streamlit_chat import message
from cogagent.toolkits.dialog_gossip_sticker_toolkit import DialogGossipStickerToolkit
import torch
from cogagent.utils.io_utils import load_file_path_yaml

# # *******************************Construct Web Pages*******************************

st.title('CogAgent')

st.markdown('''
:green[**_A MULTIMODAL, KNOWLEDGABLE AND CONTROLLABLE TOOLKIT FOR BUILDING CONVERSATIONAL AGENTS_**]
''')
st.header("MULTIMODAL RESPONSE STICKER CHINESE")



st.warning('''
This module is mainly used to reply both texts and stickers.
''')


# st.warning('''
# **This module is a dialogue based on unstructured text knowledge, which can conduct multiple rounds of dialogue. Write Exit to stop.**
# ''')


# # *******************************Loading Knowledge Grounded Agent*******************************
config = load_file_path_yaml("./config.yaml")
@st.cache_resource
def load_sticker_agent():
    agent = DialogGossipStickerToolkit(
        dataset_name="ChineseGossipDialog",
        model_name="ChineseGossipDialog",
        # vocab_path="/data/mentianyi/code/CogNLP/datapath/gossip_dialog/chinese_gossip_dialog/raw_data/vocab.txt",
        # model_path="/data/mentianyi/code/CogNLP/datapath/gossip_dialog/chinese_gossip_dialog/raw_data",
        file_or_model="file",
        sticker_dataset_name="Mod",
        sticker_model_name="StickerDialog",
        # sticker_model_path="/data/mentianyi/code/CogNLP/datapath/mm_dialog/mod/experimental_result/final--2023-01-17--14-00-56.81/model/checkpoint-780000/models.pt",
        language="chinese",
        max_history_len=3,
        generate_max_len=20,
        select_id=0,
        # id2img_path="/data/mentianyi/code/CogNLP/datapath/mm_dialog/mod/raw_data/id2img.json",
        # image_path="/data/mentianyi/code/CogNLP/datapath/mm_dialog/mod/raw_data/meme_set")
        **config["sticker"],
    )

    return agent


agent = load_sticker_agent()


# # *******************************call the openqa interface*******************************


def reset_dialogue():
    del st.session_state["generated"]
    del st.session_state["image"]
    del st.session_state["past"]
    del st.session_state["past_token"]
    del st.session_state["sticker"]
    del st.session_state["user_input_history"]
    st.session_state["user_input"] = ""



if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'image' not in st.session_state:
    st.session_state['image'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'past_token' not in st.session_state:
    st.session_state['past_token'] = []
if 'user_input_history' not in st.session_state:
    st.session_state["user_input_history"] = []
if 'user_input' not in st.session_state:
    st.session_state["user_input"] = ""

if st.session_state["user_input"]:
    output = st.session_state["user_input"]

    sentence, image, st.session_state['past'], st.session_state['past_token'] = agent.infer_one(
        user_sentence=st.session_state["user_input"],
        dialogue_history=st.session_state['past'],
        dialogue_history_token=st.session_state['past_token'])
    st.session_state["generated"].append(sentence)
    st.session_state["image"].append(image)
    st.session_state["user_input_history"].append(output)
if st.session_state['generated']:
    for i in range(0,len(st.session_state['generated'])):
        message(st.session_state['user_input_history'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))
        st.image(st.session_state["image"][i],width=150)

user_input = st.text_input("Talk to CogAgent!", key='user_input')

btn = st.button("Restart Dialogue", key='reset', on_click=reset_dialogue)
