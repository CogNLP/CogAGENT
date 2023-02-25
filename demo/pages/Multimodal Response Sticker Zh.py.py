import sys

sys.path.append('/data/mentianyi/code/CogNLP')
import streamlit as st
from streamlit_chat import message
from cogagent.toolkits.dialog_gossip_sticker_toolkit import DialogGossipStickerToolkit
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline

# # *******************************Construct Web Pages*******************************

st.title('CogAgent')

st.markdown('''
:green[**_Sticker_English_**]
''')
st.header("Sticker_English")
st.sidebar.markdown("Sticker_English")


# st.warning('''
# **This module is a dialogue based on unstructured text knowledge, which can conduct multiple rounds of dialogue. Write Exit to stop.**
# ''')


# # *******************************Loading Knowledge Grounded Agent*******************************

@st.cache_resource
def load_sticker_agent():
    agent = DialogGossipStickerToolkit(
        dataset_name="ChineseGossipDialog",
        model_name="ChineseGossipDialog",
        vocab_path="/data/mentianyi/code/CogNLP/datapath/gossip_dialog/chinese_gossip_dialog/raw_data/vocab.txt",
        model_path="/data/mentianyi/code/CogNLP/datapath/gossip_dialog/chinese_gossip_dialog/raw_data",
        file_or_model="file",
        sticker_dataset_name="Mod",
        sticker_model_name="StickerDialog",
        sticker_model_path="/data/mentianyi/code/CogNLP/datapath/mm_dialog/mod/experimental_result/final--2023-01-17--14-00-56.81/model/checkpoint-780000/models.pt",
        language="chinese",
        max_history_len=3,
        generate_max_len=20,
        select_id=0,
        id2img_path="/data/mentianyi/code/CogNLP/datapath/mm_dialog/mod/raw_data/id2img.json",
        image_path="/data/mentianyi/code/CogNLP/datapath/mm_dialog/mod/raw_data/meme_set")

    en_zh_mode_name = 'Helsinki-NLP/opus-mt-en-zh'
    en_zh_model = AutoModelWithLMHead.from_pretrained(en_zh_mode_name)
    en_zh_tokenizer = AutoTokenizer.from_pretrained(en_zh_mode_name)
    en_zh_translation = pipeline("translation_en_to_zh", model=en_zh_model, tokenizer=en_zh_tokenizer)

    zh_en_mode_name = 'Helsinki-NLP/opus-mt-zh-en'
    zh_en_model = AutoModelWithLMHead.from_pretrained(zh_en_mode_name)
    zh_en_tokenizer = AutoTokenizer.from_pretrained(zh_en_mode_name)
    zh_en_translation = pipeline("translation_zh_to_en", model=zh_en_model, tokenizer=zh_en_tokenizer)
    return en_zh_translation, zh_en_translation, agent


en_zh_translation, zh_en_translation, agent = load_sticker_agent()


# # *******************************call the openqa interface*******************************


def reset_dialogue():
    del st.session_state["generated"]
    del st.session_state["image"]
    del st.session_state["past"]
    del st.session_state["past_token"]
    del st.session_state["sticker"]
    del st.session_state["user_input_history"]
    st.session_state["user_input"] = ""


user_input = st.text_input("Talk to CogAgent!", key='user_input')

btn = st.button("Restart Dialogue", key='reset', on_click=reset_dialogue)

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
    translated_user_input = en_zh_translation(st.session_state["user_input"], max_length=400)[0]["translation_text"]

    sentence, image, st.session_state['past'], st.session_state['past_token'] = agent.infer_one(
        user_sentence=translated_user_input,
        dialogue_history=st.session_state['past'],
        dialogue_history_token=st.session_state['past_token'])
    print(translated_user_input,sentence)
    st.session_state["generated"].append(zh_en_translation(sentence, max_length=400)[0]["translation_text"])
    st.session_state["image"].append(image)
    st.session_state["user_input_history"].append(output)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        st.image(st.session_state["image"][i], width=150)
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['user_input_history'][i], is_user=True, key=str(i) + '_user')



