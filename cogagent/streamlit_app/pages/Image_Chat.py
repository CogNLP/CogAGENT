import sys,os,json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import streamlit as st
from streamlit_chat import message
from PIL import Image
from io import BytesIO
import numpy as np
from cogagent.toolkits.projects.image_chat.image_chat import setup_interactive,inference
# # 加载图像的函数
# def load_an_image(image):
#     img = Image.open(image)
#     return img
st.title('CogAgent')
# st.markdown('Streamlit is **_really_ cool**.')
# st.markdown('This text is :red[colored red], and this is **:blue[colored]** and bold.')
# st.markdown(":green[$\sqrt{x^2+y^2}=1$] is a Pythagorean identity. :pencil:")
st.markdown('''
:green[**_A KNOWLEDGE-ENHANCED TEXT REPRESENTATION TOOLKIT FOR NATURAL LANGUAGE UNDERSTANDING_**]
''')

st.header("Image Chat")

st.warning('''
**This module is mainly used to talk about the pictures uploaded by users, which can conduct multiple rounds of dialogue.
The user uploads an image and carries out multiple rounds of dialogue on the image.Write Exit to stop.**
''')
 # # *******************************Loading Knowledge Grounded Agent*******************************

@st.cache_resource
# def load_image_chat_agent():
    # setup_interactive()
    # return agent

# load_image_chat_agent()
# agent = load_vqa_agent()


# # *******************************call the openqa interface*******************************


def reset_dialogue():
    st.session_state["generated"] = []
    st.session_state["dialog_history"] = []
    st.session_state["text"] = ""
    st.session_state["personality"] = "Adventurous"


import base64
from io import BytesIO
def image_to_base64(image):
    # with open(path, 'rb') as img:
    #     # 使用base64进行编码
    #     b64encode = base64.b64encode(img.read())
    #     s = b64encode.decode()
    #     b64_encode = 'data:image/jpeg;base64,%s' % s
    #     # 返回base64编码字符串
    #     return b64_encode
    img_io = BytesIO()
    image.save(img_io, 'JPEG')
    img_byte = img_io.getvalue()
    img_base64 = base64.b64encode(img_byte)
    img_base64_str = img_base64.decode('utf-8')
    return img_base64_str

def load_an_image(image):
    img = Image.open(image)
    return img

# setup_interactive()
image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
if image_file is not None:
    st.image(load_an_image(image_file), width=250)
per_path='/data/zhaojingxuan/zjxcode/CogAgent/cogagent/toolkits/projects/image_chat/personalities.json'
with open(per_path,"rb") as f:
    pers = json.load(f)
per_list = []
for p in pers['positive']:
    per_list.append(p)
for p in pers['neutral']:
    per_list.append(p) 
for p in pers['negative']:
    per_list.append(p)
personality = st.selectbox("Select a personality",per_list, key='personality')
text = st.text_input("Talk to CogAgent!", key='text')
btn = st.button("Restart Dialogue", key='reset', on_click=reset_dialogue)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'dialog_history' not in st.session_state:
    st.session_state["dialog_history"] = []

if st.session_state["text"]:
    output = st.session_state["text"]
    personality = st.session_state["personality"]
    # data = {
    #         'image':[load_an_image(image_file)],
    #         'personality': [personality], 
    #         "text": [output]}
    # st.info(image_to_base64(Image.open(image_file)))
    base64Str = r"data:image/jpg;base64," + image_to_base64(Image.open(image_file))
    sentence = inference(e_image=base64Str,personality=personality,text=output)
    # sentence = "This is a picture about: " + infer_dict["caption"] + ". \t" + "I think the answer is: " + infer_dict[
    #     "pred_answer"] + ". \t" + "Because: " + infer_dict["reason"]
    # st.info(sentence)
    st.session_state["generated"].append(sentence)
    st.session_state["dialog_history"].append(output)
    # st.session_state["dialog_history"].append(personality)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['dialog_history'][i], is_user=True, key=str(i) + '_user')  
