import streamlit as st
from cogagent.utils.log_utils import logger
from cogagent.utils.io_utils import load_file_path_yaml
from cogagent.toolkits.openqa_toolkit import OpenqaAgent
import torch

# # *******************************Construct Web Pages*******************************

st.title('CogAgent')

st.markdown('''
:green[**_A MULTIMODAL, KNOWLEDGEABLE AND CONTROLLABLE TOOLKIT FOR BUILDING CONVERSATIONAL AGENTS_**]
''')
st.header("Open Domain Question Answering")
st.sidebar.markdown("Open Domain Question Answering")
# st.warning('''
# **This module is a question and answer in the open domain, which is a single round of dialogue.
# Users ask questions, and this module gives answers and retrieves relevant articles from Wikipedia. Write Exit to stop.**
# ''')

# choice = st.selectbox('Question', ['Select an example or input text', 'Which part of earth is covered with water?',
#                                    'Who proved that genes are located on chromosomes?',
#                                    'Where will the summer olympics be held in 2020?',
#                                    'Who has the largest contract in the NBA?',
#                                    'Who has the most points in nba finals history?',
#                                    'Who died in Harry Potter and the Half-Blood Prince?'])
# if choice == 'Select an example or input text':
#     question = st.text_input('Input text here', '')
# else:
#     question = st.text_input('Input text here', choice)
question = st.text_input("Input your question here:")
n_doc = st.slider('Please select the number of relevant articles retrieved from Wikipedia', 1, 10)

submit = st.button('SUBMIT')


# # *******************************Loading OpenQA Agent*******************************
config = load_file_path_yaml("./config.yaml")
@st.cache_resource
def load_openqa_agent():
    logger.info("Loading openqa agent...")
    agent = OpenqaAgent(
        bert_model="bert-base-uncased",
        device=torch.device("cuda:9"),
        **config["openqa"],
    )
    logger.info("Loading openqa agent Finished.")
    return agent


agent = load_openqa_agent()

# # *******************************call the openqa interface*******************************
if submit:
    if question != 'Exit':
        wiki_psgs = agent.search_wiki(question)
        dataloader = agent.construct_dataloader(question, wiki_psgs)
        _, pred_text = agent.predict(dataloader)

        st.write("Answer")
        st.success(pred_text)
        st.write("Relevant Articles")
        for i in range(n_doc):
            st.success(wiki_psgs[i][1])
    else:
        st.info('Thanks')
