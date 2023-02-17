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

st.header("Mutimodal Retrieval")

st.warning('''
**This module is a multimodal retrieval dialogue, which is a single round of dialogue. When users ask questions, 
the module will divide the answers into three types: text, picture and table and show them to users.Write Exit to stop**
''')

choice = st.selectbox('Question',
                      ['Select an example or input text','Which city features a green copper statue of a woman holding a torch?'])
if choice == 'Select an example or input text':
    question = st.text_input('Input text here', '')
else:
    question = st.text_input('Input text here', choice)
       
submit = st.button('SUBMIT')

if submit:
    if question != 'Exit':
        ## return
        result = {
                "pred_text": 'New York City.', 
                "retrieval_type": 'image', 
                "retrieval_content": 'The Statue of Liberty, also known as the Statue of Liberty National Monument, is located on Liberty Island, New York, USA', 
                "retrieval_image": '/data/yangcheng/MMCoQA/data/final_dataset_images/2b79346aa78121aaaf7e0540983d21ea.jpg'  }

        if question == 'Which city features a green copper statue of a woman holding a torch?':
            # st.json(result)
            st.write('Answer')
            st.success(result['pred_text']) 
            st.success(result['retrieval_content']) 
            if(result['retrieval_type']=='image'):
                st.image('https://img.mp.itc.cn/q_70,c_zoom,w_640/upload/20161003/92511364a7dd4ac29490c1b8c8cbdab2_th.jpg')
        #title1 = st.text_input('Question', 'Input more')
    else:
        st.info('Thanks')
        



