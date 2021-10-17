import streamlit as st
import cv2 
import numpy as np 
from keras.models import load_model

def app():      
    model = load_model("need.h5")
    CLASS_NAMES = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
    st.title("Garbage Classification")
    st.write("Welcome")
    st.image('1.png')
    st.write("Garbage Recycling is a key aspect of preserving our environment. To make the recycling process possible/easier, the garbage must be sorted to groups that have similar recycling process. This we help you to classify garbage into a few classes (2 to 6 classes at most). Having the ability to sort the household garbage into more classes can result in dramatically increasing the percentage of the recycled garbage.")
    img1 = st.file_uploader("Choose an image...", type="jpg")
    submit = st.button('Predict')
    if submit:
        if img1 is not None:
            file_bytes = np.asarray(bytearray(img1.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            st.image(opencv_image, channels="BGR")
            opencv_image = cv2.resize(opencv_image, (224,224))
            opencv_image.shape = (1,224,224,3)
            pred = model.predict(opencv_image)
            print(np.argmax(pred))
            st.title(str(CLASS_NAMES[np.argmax(pred)]))