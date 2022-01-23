import streamlit as st
from tensorflow import keras
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
from tensorflow.keras.preprocessing.image import smart_resize
from mtcnn.mtcnn import MTCNN
from PIL import Image, ImageDraw, ImageFont

def load_model():
    model = keras.models.load_model("mask_detector_vgg19.h5")
    return model

def load_face_detector():
    detector = MTCNN()
    return detector

# @st.cache
# def resize(img):
#     img = Image.open(img)
#     img_ary = img_to_array(img)
#     resized_img = array_to_img(smart_resize(img_ary, (256, 256)))
#     return resized_img


def mask_detect(img, model, detector):
    img_ary = img_to_array(img)
    faces = detector.detect_faces(img_ary)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 20)
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
        
        draw.rectangle([(x1,y1),(x2,y2)], outline=color, width=2)
        draw.text((x1,y1-20), txt, fill=color, font=font)
    # st.pyplot(fig)
    # plt.show()
    return img.resize((512,512))

if __name__ == '__main__':
    st.title('Welcome to Face-Mask Detector!')
    img = st.file_uploader('Please Upload an Image', type=['png', 'jpg', 'jpeg'])
    model = load_model()
    detector = load_face_detector()
    if img:
        st.title('Original image uploaded:')
        img = Image.open(img)
        st.image(img.resize((512,512)))   
        st.title('Result:')
        labelled_img = mask_detect(img, model, detector)
        st.image(labelled_img)