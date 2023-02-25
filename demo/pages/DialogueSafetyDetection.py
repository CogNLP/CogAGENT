import streamlit as st
from cogagent.utils.log_utils import logger
from cogagent.toolkits.dia_safety_toolkit import DialogeSafetyDetectionToolkit
import torch

# # *******************************Construct Web Pages*******************************

st.title('CogAgent')

st.markdown('''
:green[**_A MULTIMODAL, KNOWLEDGEABLE AND CONTROLLABLE TOOLKIT FOR BUILDING CONVERSATIONAL AGENTS_**]
''')

st.header("Dialogue Safety Detection")
st.sidebar.markdown("Dialogue Safety Detection")
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
context = st.text_input("Input your utterance here:")
response = st.text_input("Input the agent's response here:")
# n_doc = st.slider('Please select the number of relevant articles retrieved from Wikipedia', 1, 10)

submit = st.button('SUBMIT')


# # *******************************Loading OpenQA Agent*******************************

@st.cache_resource
def load_dia_safety_detection_toolkit():
    logger.info("Loading open dialogue safety detection toolkit...")
    toolkit = DialogeSafetyDetectionToolkit(
        plm_name='roberta-base',
        classifier_path='/data/hongbang/CogAGENT/datapath/dialogue_safety/DiaSafety/raw_data/classifiers',
        device=torch.device("cuda:1"),
    )
    logger.info("Loading Finished.")
    return toolkit


toolkit = load_dia_safety_detection_toolkit()

# # *******************************call the openqa interface*******************************
if submit:
    safe,category = toolkit.run(context,response)
    if safe:
        st.success("Safe!")
        # st.write("Safety:",safe,"  Category:",category)
    else:
        st.warning("Unsafe Behaviour!")
        st.write("Unsafe Category:",category)
        # msg = "Unsafe!Category:"+category
        # st.write(msg)
        # st.warning(msg)
    # print("End")

    # if question != 'Exit':
    #     wiki_psgs = agent.search_wiki(question)
    #     dataloader = agent.construct_dataloader(question, wiki_psgs)
    #     _, pred_text = agent.predict(dataloader)
    #
    #     st.write("Answer")
    #     st.success(pred_text)
    #     st.write("Relevant Articles")
    #     for i in range(n_doc):
    #         st.success(wiki_psgs[i][1])
    # else:
    #     st.info('Thanks')
