#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import app5
import app2
import app3
import app4
import app1
PAGES = {
    "Garbage": app5 ,
    "Mental Health": app2 ,
    "Health" : app3 , 
    "Covid19": app4 ,
    "Plant_Disease" :app1
}
st.sidebar.title('Arogya ')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()