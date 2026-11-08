import os
import cv2 as cv2
import mediapipe as mp
import numpy as np
import pickle

cap = cv2.VideoCapture(0)
if cap is None or not cap.isOpened():
    print("bad camera")
    exit(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

PATH = ".\\data\\closed\\"

frame_no = 1

data = []

while frame_no < 300:
    ret, frame = cap.read()

    res = hands.process(frame)
    if res.multi_hand_landmarks:
        X = []
        Y = []
        temp = []
        for hand_landmarks in res.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                X.append(hand_landmarks.landmark[i].x)
                Y.append(hand_landmarks.landmark[i].y)

            x_min = min(X)
            y_min = min(Y)

            for i in range(len(hand_landmarks.landmark)):
                temp.append(X[i] - x_min)
                temp.append(Y[i] - y_min)

            data.append(temp)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                      mp_style.get_default_hand_landmarks_style(),
                                      mp_style.get_default_hand_connections_style())
            frame_no += 1

    cv2.imshow('frame', frame)
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()

file = open('closed.pickle', 'wb')
pickle.dump({"closed":data}, file)
file.close()
