import sys

sys.path.append('/data/mentianyi/code/CogNLP')
import streamlit as st
from streamlit_chat import message
from cogagent.toolkits.dialog_vqa_toolkit import DialogVQAToolkit
from PIL import Image
from io import BytesIO
import torch
from cogagent.utils.io_utils import load_file_path_yaml
# os.environ["OPENAI_API_KEY"] = <"your_api_key">
# os.environ['OPENAI_API_KEY']  = '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'
# # *******************************Construct Web Pages*******************************

st.title('CogAgent')

st.markdown('''
:green[**_A MULTIMODAL, KNOWLEDGABLE AND CONTROLLABLE TOOLKIT FOR BUILDING CONVERSATIONAL AGENTS_**]
''')
st.header("MULTIMODAL ENHANCEMENT KNOWLEDGE-BASED VQA")



st.warning('''
**This module is mainly used to ask questions about pictures uploaded by users. We use GPT-3 as external knowledge base. It can answer some commonsense questions about the picture.**
''')


# # *******************************Loading Knowledge Grounded Agent*******************************
config = load_file_path_yaml("./config.yaml")
@st.cache_resource
def load_vqa_agent():
    agent = DialogVQAToolkit(
        dataset_name="OKVQA",
        model_name="pica",
        # data_path="/data/hongbang/.cogagent/okvqa",
        language="english",
        **config["vqa"]
    )
    return agent



agent = load_vqa_agent()


# # *******************************call the openqa interface*******************************


def reset_dialogue():
    del st.session_state["generated"]
    del st.session_state["user_input_history"]
    st.session_state["user_input"] = ""


def load_an_image(image):
    img = Image.open(image)
    return img


image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
if image_file is not None:
    st.image(load_an_image(image_file), width=600)


if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'user_input_history' not in st.session_state:
    st.session_state["user_input_history"] = []
if 'user_input' not in st.session_state:
    st.session_state["user_input"] = ''

if st.session_state["user_input"]:
    output = st.session_state["user_input"]
    raw_dict = {"image_url": load_an_image(image_file),
                "question": output}

    infer_dict = agent.infer_one(raw_dict=raw_dict, image_str=False)
    sentence = "This is a picture about: " + infer_dict["caption"] + ". \t" + "I think the answer is: " + infer_dict[
        "pred_answer"] + "." + " Because: " + infer_dict["reason"]
    st.session_state["generated"].append(sentence)
    st.session_state["user_input_history"].append(output)
if st.session_state['generated']:
    for i in range(0,len(st.session_state['generated']) ):
        message(st.session_state['user_input_history'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))


user_input = st.text_input("Talk to CogAgent!", key='user_input')
btn = st.button("Restart Dialogue", key='reset', on_click=reset_dialogue)
