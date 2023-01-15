import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

import streamlit as st
import tempfile
import time
from PIL import Image
import pickle 
import pandas as pd
import pyttsx3  # for alarm sounds
import threading


# Load model
model = pickle.load(open('body_language.pkl','rb'))
    
    
    


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    

    
    
    def recv(self, frame):

        # Drawing Helpers - to draw the keypoints and lines on video feed
        mp_drawing = mp.solutions.drawing_utils 
        # Holistic pipeline integrates separate models for pose, face and hand components
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
        img = frame.to_ndarray(format="bgr24")

        ## Face Mesh
        with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:

            img = cv2.flip(img,1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            #FRAME_WINDOW = st.image([])
            #FRAME_WINDOW.image(img)

            img.flags.writeable = False    

            # Make Detections
            results = holistic.process(img)

             # Recolor image back to BGR for rendering
            img.flags.writeable = True   
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


            # Export coordinates
            try:
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
                body_language_prob = model.predict_proba(X)[0]
                print(body_language_class, body_language_prob)

                # alarm properties for alert
                alarm_sound = pyttsx3.init()
                voices = alarm_sound.getProperty('voices')
                alarm_sound.setProperty('voice', voices[0].id)
                alarm_sound.setProperty('rate', 150)

                # Play alarm based on 'class'
                if body_language_class == 'Not Urgent':
                    alarm_sound.say("Not urgent")
                    alarm_sound.runAndWait()
                    if alarm_sound._inLoop:
                        alarm_sound.endLoop()
                    # alarm = threading.Thread(target=not_urgent_alarm, args=(alarm_sound,))
                    # alarm.start()

                elif body_language_class == 'Urgent':
                    alarm_sound.say("Immediate attention needed")
                    alarm_sound.runAndWait()
                    if alarm_sound._inLoop:
                        alarm_sound.endLoop()
                    # alarm = threading.Thread(target=urgent_alarm, args=(alarm_sound,))
                    # alarm.start()

                    #break the loop 
                    # break  -- it will go straight to the the end. to think of what to do in the next frame. or manually terminate for demo. 

                ##### terminate window? 
                ### to add another criteria for alarm here ###

                # Grab shoulder coords
                coords = tuple(np.multiply(
                                np.array(
                                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x, 
                                     results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y))
                            , [640,480]).astype(int))


                # Dashboard
                # kpil_text.write(f"<h1 style='text-align: center; color:red;'>{body_language_class}</h1>", unsafe_allow_html=True)
                
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            except:
                pass

        
        
        
        
kpil, kpil2, kpil3 = st.columns(3)

with kpil:
    st.markdown('**Body Language Class**')
    kpil_text = st.markdown('0')

st.markdown('<hr/>', unsafe_allow_html=True)

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)