import streamlit as st
from PIL import Image
#import torch
import numpy as np

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

import cv2
import streamlit as st
import mediapipe as mp
import cv2 as cv
import numpy as np
import tempfile
import time
import pickle
import pandas as pd
import mediapipe as mp 
import pyttsx3  # for alarm sounds
import threading


device = 'cpu'
if not hasattr(st, 'classifier'):
    st.model = pickle.load(open('body_language.pkl','rb'))
    #st.model = torch.hub.load('ultralytics/yolov5', 'yolov5s',  _verbose=False)


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class VideoProcessor:
    def recv(self, frame):
        
        try:
        
        
            img = frame.to_ndarray(format="bgr24")

            # vision processing
            # flipped = img[:, ::-1, :]

            # model processing
            # im_pil = Image.fromarray(flipped)
            img = cv2.flip(img,1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img.flags.writeable = False  
            # results = st.model(im_pil, size=112)

            mp_holistic = mp.solutions.holistic
            holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)
            results = holistic.process(img)


             # Recolor image back to BGR for rendering
            img.flags.writeable = True   
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Extract Right Hand landmarks
            rhand = results.right_hand_landmarks.landmark
            rhand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in rhand]).flatten())

            # Extract Left Hand landmarks
            lhand = results.left_hand_landmarks.landmark
            lhand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in lhand]).flatten())


            # Concate rows
            row = pose_row + rhand_row + lhand_row

            # Make Detections
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            # body_language_prob = model.predict_proba(X)[0]
            # print(body_language_class, body_language_prob)


            bbox_img = np.array(body_language_class)
            # bbox_img = np.array(results.render()[0])

            return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")

        except:
            pass
        
        
        

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
