import streamlit as st
from streamlit_chat import message
from cogagent.utils.log_utils import logger
from cogagent.toolkits.knowledge_grounded_dialogue_toolkit import KnowledgeGroundedConversationAgent
import torch

# # *******************************Construct Web Pages*******************************

st.title('CogAgent')

st.markdown('''
:green[**_A MULTIMODAL, KNOWLEDGEABLE AND CONTROLLABLE TOOLKIT FOR BUILDING CONVERSATIONAL AGENTS_**]
''')
st.header("Knowledge Grounded Dialogue")
st.sidebar.markdown("Knowledge Grounded Dialogue")


# st.warning('''
# **This module is a dialogue based on unstructured text knowledge, which can conduct multiple rounds of dialogue. Write Exit to stop.**
# ''')

# # *******************************Loading Knowledge Grounded Agent*******************************

@st.cache_resource
def load_knowledge_grounded_agent():
    logger.info("Loading open qa agent...")
    agent = KnowledgeGroundedConversationAgent(
        bert_model=None,
        model_path='/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/experimental_result/run_diffks_wow_lr1e-4--2022-11-16--01-01-39.05/best_model/checkpoint-376000/models.pt',
        vocabulary_path='/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/cache/wow_vocab.pkl',
        device=torch.device("cuda:0"),
        debug=True,
    )
    logger.info("Loading Finished.")
    return agent


agent = load_knowledge_grounded_agent()


# # *******************************call the openqa interface*******************************


def reset_dialogue():
    del st.session_state["generated"]
    del st.session_state["past"]
    if 'knowledge_list' in st.session_state:
        del st.session_state["knowledge_list"]
    if 'topic' in st.session_state:
        del st.session_state["topic"]
    st.session_state["user_input"] = ""


def set_dialogue_topic(topic):
    st.session_state["topic"] = topic


if 'user_input' not in st.session_state:
    st.session_state["user_input"] = ""
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# if 'knowledge_list' not in st.session_state:
#     if "topic" in st.session_state:
#         st.session_state['knowledge_list'] = agent.get_knowledge_from_topic(topic=st.session_state["topic"])

if st.session_state["user_input"]:
    # st.session_state["user_input"]
    # st.write([st.session_state["past"][-1], st.session_state["generated"][-1]] if st.session_state["generated"] else [])
    # st.session_state["knowledge_list"]
    output = st.session_state["user_input"]
    output = agent.get_response(
        user_utterance=st.session_state["user_input"],
        chat_history=[st.session_state["past"][-1],st.session_state["generated"][-1]] if st.session_state["past"] else [],
        knowledge_list=st.session_state["knowledge_list"]
    )
    # store the output
    st.session_state['past'].append(st.session_state["user_input"])
    st.session_state['generated'].append(output)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))


user_input = st.text_input("Talk to CogAgent!", key='user_input')

btn = st.button("Restart Dialogue", key='reset', on_click=reset_dialogue)

# generate response:
if 'topic' not in st.session_state and not user_input:
    # topic = st.selectbox(
    #     'Select one topic you would like to talk.',
    #     agent.get_topic_candidates(),key='topic')
    # topic = st.radio(
    #     'Select one topic you would like to talk.',
    #     ["Choosing one topic from below."]+agent.get_topic_candidates(),key='topic')
    topic = st.radio(
        'Select one topic you would like to talk.',
        ["Choosing one topic from below."]+['Science fiction','Dating','Piano','New York City','Open Relationship'],key='topic')
    # topic = st.text_input("Input the topic")
    # submit = st.button("Submit")
    # if submit:import streamlit as st
    # from cogagent.utils.log_utils import logger
    # from cogagent.toolkits.openqa_toolkit import OpenqaAgent
    # import torch
    #
    # # # *******************************Construct Web Pages*******************************
    #
    # st.title('CogAgent')
    #
    # st.markdown('''
    # :green[**_A KNOWLEDGE-ENHANCED TEXT REPRESENTATION TOOLKIT FOR NATURAL LANGUAGE UNDERSTANDING_**]
    # ''')
    #
    # st.header("Open Domain Question Answering")
    # st.sidebar.markdown("Open Domain Question Answering")
    # # st.warning('''
    # # **This module is a question and answer in the open domain, which is a single round of dialogue.
    # # Users ask questions, and this module gives answers and retrieves relevant articles from Wikipedia. Write Exit to stop.**
    # # ''')
    #
    # # choice = st.selectbox('Question', ['Select an example or input text', 'Which part of earth is covered with water?',
    # #                                    'Who proved that genes are located on chromosomes?',
    # #                                    'Where will the summer olympics be held in 2020?',
    # #                                    'Who has the largest contract in the NBA?',
    # #                                    'Who has the most points in nba finals history?',
    # #                                    'Who died in Harry Potter and the Half-Blood Prince?'])
    # # if choice == 'Select an example or input text':
    # #     question = st.text_input('Input text here', '')
    # # else:
    # #     question = st.text_input('Input text here', choice)
    # question = st.text_input("Input your question here:")
    # n_doc = st.slider('Please select the number of relevant articles retrieved from Wikipedia', 1, 10)
    #
    # submit = st.button('SUBMIT')
    #
    #
    # # # *******************************Loading OpenQA Agent*******************************
    #
    # @st.cache_resource
    # def load_openqa_agent():
    #     logger.info("Loading open qa agent...")
    #     agent = OpenqaAgent(
    #         bert_model="bert-base-uncased",
    #         model_path='/data/hongbang/CogKTR/datapath/question_answering/NaturalQuestions/raw_data/bert-base-mrc-openqa.pt',
    #         device=torch.device("cuda:9"),
    #         debug=False,
    #     )
    #     logger.info("Loading Finished.")
    #     return agent
    #
    #
    # agent = load_openqa_agent()
    #
    # # # *******************************call the openqa interface*******************************
    # if submit:
    #     if question != 'Exit':
    #         wiki_psgs = agent.search_wiki(question)
    #         dataloader = agent.construct_dataloader(question, wiki_psgs)
    #         _, pred_text = agent.predict(dataloader)
    #
    #         st.write("Answer")
    #         st.success(pred_text)
    #         st.write("Relevant Articles")
    #         for i in range(n_doc):
    #             st.success(wiki_psgs[i][1])
    #     else:
    #         st.info('Thanks')
    #     st.session_state["topic"] = topic

if 'topic' in st.session_state and st.session_state["topic"] != 'Choosing one topic from below.':
    st.write("Dialogue Topic:", st.session_state.topic)
    st.session_state['knowledge_list'] = agent.get_knowledge_from_topic(topic=st.session_state["topic"])