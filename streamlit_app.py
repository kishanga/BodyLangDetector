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
        img = frame.to_ndarray(format="bgr24")
        
        # vision processing
        flipped = img[:, ::-1, :]

        # model processing
        im_pil = Image.fromarray(flipped)
        # results = st.model(im_pil, size=112)
        
        results = holistic.process(im_pil)
        
        # body_language_class = model.predict(X)[0]
        #body_language_prob = model.predict_proba(X)[0]
        # print(body_language_class, body_language_prob)
        
        
        bbox_img = np.array(results.render()[0])

        return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
