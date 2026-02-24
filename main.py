import streamlit as st
import cv2
from face_landmark import face_landmark
from hand_landmark import hand_landmark
from hand_mouse import hand_mouse

st.title("Yamen's Computer Vision App")
st.write("This app demonstrates the use of Mediapipe's face and hand landmark detection models in a Streamlit application. You can choose to run the face landmark detection, hand landmark detection, or hand mouse control features by clicking the corresponding buttons below.")

tab1, tab2, tab3 = st.tabs(["Face Landmark Detection", "Hand Landmark Detection", "Hand Mouse Control"])

vedio_placeholder = st.empty()

with tab1:
    if st.button("Run Face Landmark Detection"):
        st.session_state['run_face_landmark'] = True
        face_landmark(vedio_placeholder)
    elif st.button("Stop Face Landmark Detection"):
        st.session_state['run_face_landmark'] = False

with tab2:
    if st.button("Run Hand Landmark Detection"):
        st.session_state['run_hand_landmark'] = True
        hand_landmark(vedio_placeholder)
    elif st.button("Stop Hand Landmark Detection"):
        st.session_state['run_hand_landmark'] = False

with tab3:
    if st.button("Run Hand Mouse Control"):
        st.session_state['run_mouse'] = True
        hand_mouse(vedio_placeholder)
    elif st.button("Stop Hand Mouse Control"):
        st.session_state['run_mouse'] = False


st.write("YamenRM - 2026")
st.write("This app is built using Streamlit , Mediapipe, OpenCV, and PyAutoGUI. It demonstrates real-time face and hand landmark detection, as well as hand gesture-based mouse control. Feel free to explore the different features and have fun experimenting with computer vision!")

