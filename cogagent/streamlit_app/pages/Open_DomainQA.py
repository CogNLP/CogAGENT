import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
# import cv2 # 计算机视觉

# 加载图像的函数
# def load_an_image(image):
#     img = Image.open(image)
#     return img

######################################################
st.title('CogAgent')

st.markdown('''
:green[**_A KNOWLEDGE-ENHANCED TEXT REPRESENTATION TOOLKIT FOR NATURAL LANGUAGE UNDERSTANDING_**]
''')

st.header("Open Domain Question Answering")

st.warning('''
**This module is a question and answer in the open domain, which is a single round of dialogue. 
Users ask questions, and this module gives answers and retrieves relevant articles from Wikipedia. Write Exit to stop.**
''')

# 图片文件上传器
# image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
# if image_file is not None:
#     st.image(load_an_image(image_file), width=250)

choice = st.selectbox('Question',['Select an example or input text','Which part of earth is covered with water?',
'Who proved that genes are located on chromosomes?',
'Where will the summer olympics be held in 2020?',
'Who has the largest contract in the NBA?',
'Who has the most points in nba finals history?' ,
'Who died in Harry Potter and the Half-Blood Prince?'])
if choice == 'Select an example or input text':
    question = st.text_input('Input text here', '')
else:
    question = st.text_input('Input text here', choice)
n_doc = st.slider('Please select the number of relevant articles retrieved from Wikipedia',0,10) 
       
submit = st.button('SUBMIT')

if submit:
    if question != 'Exit':
        ## return
        # psgs = []
        # for i in range(n_doc):
        #     k = 'passages_content_' + str(i)
        #     psgs.append(k)
        result = {
        "pred_answer": '71 percent', 
        # "wiki_psgs": psgs
        }

        if question == 'Which part of earth is covered with water?':
            # st.json(result)
            st.write('Answer')
            st.success(result['pred_answer'])    
            st.write('Relevant Articles')
            for i in range(n_doc):
                k = 'passages_content_' + str(i)
                st.success(k)  
        #title1 = st.text_input('Question', 'Input more')
    else:
        st.info('Thanks')
        



