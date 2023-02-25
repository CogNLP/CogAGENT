import streamlit as st
from streamlit_chat import message
from cogagent.utils.log_utils import logger
from cogagent.toolkits.kemp_toolkit import KempToolkit
import torch

# # *******************************Construct Web Pages*******************************

st.title('CogAgent')

st.markdown('''
:green[**_A MULTIMODAL, KNOWLEDGEABLE AND CONTROLLABLE TOOLKIT FOR BUILDING CONVERSATIONAL AGENTS_**]
''')
st.header("Empathetic Dialogue")
st.sidebar.markdown("Empathetic Dialogue")


# # *******************************Loading Empathetic Dialogue Agent*******************************

@st.cache_resource
def load_empathetic_dialogue_agent():
    logger.info("Loading empathetic dialogue agent...")
    toolkit = KempToolkit(bert_model=None,
                          model_path='/data/hongbang/CogAGENT/datapath/kemp/experimental_result/simple_test--2023-01-30--17-28-17.80/model/checkpoint-50000/models.pt',
                          vocabulary_path='/data/hongbang/CogAGENT/datapath/kemp/raw_data/vocab.json',
                          device=torch.device("cuda:0"),
                          )

    logger.info("Loading Finished.")
    return toolkit


toolkit = load_empathetic_dialogue_agent()


# # *******************************call the openqa interface*******************************

def reset_dialogue():
    del st.session_state["generated"]
    del st.session_state["past"]
    st.session_state["user_input"] = ""


def set_dialogue_topic(topic):
    st.session_state["topic"] = topic


if 'user_input' not in st.session_state:
    st.session_state["user_input"] = ""
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

if st.session_state["user_input"]:
    output = toolkit.run([st.session_state["user_input"]])
    # store the output
    st.session_state['past'].append(st.session_state["user_input"])
    st.session_state['generated'].append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))

user_input = st.text_input("Talk to CogAgent!", key='user_input')

btn = st.button("Restart Dialogue", key='reset', on_click=reset_dialogue)
