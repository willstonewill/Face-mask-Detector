import glob
import os
import streamlit as st
from tensorflow import keras
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
from tensorflow.keras.preprocessing.image import smart_resize
from mtcnn.mtcnn import MTCNN
from PIL import Image, ImageDraw, ImageFont, ImageOps

def load_model():
    model = keras.models.load_model("Models/mask_detector_vgg19.h5")
    return model

def load_face_detector():
    detector = MTCNN()
    return detector


def mask_detect(img, model, detector):
    img_ary = img_to_array(img)
    faces = detector.detect_faces(img_ary)
    draw = ImageDraw.Draw(img)

    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1+width, y1+height

        pred = model.predict(smart_resize(img_ary[y1:y2, x1:x2], (256, 256)).reshape(1, 256, 256, 3))
        
        if pred[0][0] >= pred[0][1]:
            txt = 'No Mask'
            color = 'red'
        else:
            txt = 'Masked'
            color = 'lime'

        font = ImageFont.truetype("Fonts/arial.ttf", round(width/4))
        draw.rectangle([(x1,y1),(x2,y2)], outline=color, width=2)
 
        if y1-height/5-2 <= 0:
            draw.text((x1+2,y1), txt, fill=color, font=font)
        else:
            draw.text((x1,y1-height/5-2), txt, fill=color, font=font)

    return ImageOps.contain(img, (666, 666))

if __name__ == '__main__':
    st.title('Welcome to Face-Mask Detector!')
    img = st.file_uploader('Please Upload an Image', type=['jpg', 'jpeg'])
    model = load_model()
    detector = load_face_detector()

    d_names = []
    for name in glob.glob("Datasets/Demo_img/*"):
        d_names.append(os.path.basename(name))
    
    demo_name = st.sidebar.selectbox('Demo image', d_names)
    if not img:
        img = os.path.join("Datasets/Demo_img/", demo_name)

    st.title('Original image uploaded:')
    img = Image.open(img)
    img = ImageOps.contain(img, (666, 666))
    st.image(img)   
    st.title('Result:')
    labelled_img = mask_detect(img, model, detector)
    st.image(labelled_img)