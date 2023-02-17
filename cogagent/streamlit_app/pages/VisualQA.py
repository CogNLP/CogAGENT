import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np

# 加载图像的函数
def load_an_image(image):
    img = Image.open(image)
    return img
st.title('CogAgent')
# st.markdown('Streamlit is **_really_ cool**.')
# st.markdown('This text is :red[colored red], and this is **:blue[colored]** and bold.')
# st.markdown(":green[$\sqrt{x^2+y^2}=1$] is a Pythagorean identity. :pencil:")
st.markdown('''
:green[**_A KNOWLEDGE-ENHANCED TEXT REPRESENTATION TOOLKIT FOR NATURAL LANGUAGE UNDERSTANDING_**]
''')

st.header("Visual Question Answering")

st.warning('''
**This module is mainly used to talk about the pictures uploaded by users, which can conduct multiple rounds of dialogue.
The user uploads an image and carries out multiple rounds of dialogue on the image.Write Exit to stop.**
''')
   
l = 10  
# 最多10轮对话    
i = 0
k = 0 #让selectbox的key值唯一
# 图片文件上传器
image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
c1 = st.container()
c2 = st.container()
e1 = c2.empty()
e2 = c2.empty()

while(1):
    choice = e1.selectbox('Question',['Select an example or input text','What job often ride it?','Exit'],key = k)
    question = e2.text_input('Input text here','',key = k+100)
    if choice == 'Select an example or input text':
        question = e2.text_input('Input text here','',key = k+10)
        # break
    else:
        question = e2.text_input('Input text here',choice,key = k+10)
    k = k+1
    
    if question == '':
        break
    else:
        col1, col2 = c1.columns(2)
        with col1:
            if(i==0):
                st.image('https://img.zcool.cn/community/01d5e958158d44a84a0d304f1c3e87.png@1280w_1l_2o_100sh.png',width=50)
                # st.subheader("User")
                if image_file is not None:
                    st.image(load_an_image(image_file), width=250)
            st.info(question)

        with col2:
            if(i==0): 
                st.image('https://www.leyantech.com/themes/leyan/public/assets/images/helpOther/robot.png',width=50)
                # st.subheader("CogAgent")       
            if question != 'Exit':
                result = {
                    "pred_text": 'Police.',
                    "reason": 'The police are often in the car, and the car is often in the city, so the police car is often in the city.', 
                    "caption": 'a black motorcycle parked in a parking lot.'
                    }
                if question == 'What job often ride it?':
                    # st.json(result)
                    # st.image()
                    st.success(result['pred_text']+'The picture shows '+result['caption']+result['reason']) 
                    i = i + 1
            else:
                st.success('Thanks')
                st.stop()
                break
        
    if(i > l-1):
        st.error('The maximum number of rounds of conversation is 10!')
        break
    
    