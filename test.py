import cv2 as cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
if cap is None or not cap.isOpened():
    print("bad camera")
    exit(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

while True:
    ret, frame = cap.read()

    res = hands.process(frame)
    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                      mp_style.get_default_hand_landmarks_style(),
                                      mp_style.get_default_hand_connections_style())

    cv2.imshow('frame', frame)
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()
