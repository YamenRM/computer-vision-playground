import streamlit as st
import cv2
from face_landmark import face_landmark
from hand_landmark_mask import hand_landmark_mask
from hand_mouse import hand_mouse

st.title("Computer Vision Playground")
st.write("This app demonstrates real-time face and hand landmark mask, as well as hand gesture-based mouse control using Mediapipe and Streamlit.")

tab1, tab2, tab3 = st.tabs(["Face Landmark Detection", "Hand Landmark Mask", "Hand Mouse Control"])

vedio_placeholder = st.empty()

with tab1:
    if st.button("Run Face Landmark Detection"):
        st.session_state['run_face_landmark'] = True
        face_landmark(vedio_placeholder)
    elif st.button("Stop Face Landmark Detection"):
        st.session_state['run_face_landmark'] = False

with tab2:
    if st.button("Run Hand Landmark Mask"):
        st.session_state['run_hand_landmark_mask'] = True
        hand_landmark_mask(vedio_placeholder)
    elif st.button("Stop Hand Landmark Mask"):
        st.session_state['run_hand_landmark_mask'] = False

with tab3:
    if st.button("Run Hand Mouse Control"):
        st.session_state['run_mouse'] = True
        hand_mouse(vedio_placeholder)
    elif st.button("Stop Hand Mouse Control"):
        st.session_state['run_mouse'] = False


st.write("YamenRM - 2026")
st.write("Feel free to explore the different features and have fun experimenting with computer vision!")

