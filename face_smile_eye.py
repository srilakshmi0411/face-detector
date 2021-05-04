import cv2,time
import numpy as np


face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade=cv2.CascadeClassifier("haarcascade_smile.xml")
eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")
eyeglasses_cascade=cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

import cv2
import streamlit as st

st.title("Face Smile Eye Detector")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
num_frames =0
video=cv2.VideoCapture(0)

while run:
    check,frame=video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    for x,y,w,h in face:
        img=cv2.rectangle(frame,(x,y),(x+w,y+h),(128,128,0),3)
        smile=smile_cascade.detectMultiScale(gray,scaleFactor=1.8,minNeighbors=20)
        for x,y,w,h in smile:
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        eye=eye_cascade.detectMultiScale(gray,scaleFactor=1.8,minNeighbors=20)
        for x,y,w,h in eye:
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        eyeglasses=eyeglasses_cascade.detectMultiScale(gray,scaleFactor=1.8,minNeighbors=20)
        for x,y,w,h in eyeglasses:
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),3)


    cv2.imshow('Face,Eye,Smile Detector',frame)
    FRAME_WINDOW2.image(frame)
    key=cv2.waitKey(1)

    if key==ord('q'):
         break

else:
    st.write('Stopped')
#video.release()
#cv2.destroyAllWindows    
