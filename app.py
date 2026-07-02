from ultralytics import YOLO
import streamlit as st
import cv2
import os


st.title('Object Tracking System')
file = st.file_uploader('Upload File')

if file:
    data = file.read()
    with open('temp.mp4', 'wb') as f:
        f.write(data)

    cap = cv2.VideoCapture('temp.mp4')

    if cap:
        model = YOLO("yolo26n.pt")
        frame_destroyer = st.empty()

        ret = True
        while ret:
            ret, frame = cap.read()
            if ret:
                result = model.track(frame, persist=True)
                result_frame = result[0].plot()

                img = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                frame_destroyer.image(img)

    cap.release()
    os.remove('temp.mp4')

