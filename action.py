import cv2 as cv2
import mediapipe as mp
import numpy as np
import math
import socket
import struct

import pickle

# set up messaging server
def send_message(sock, message):
    message_length = len(message)
    header = struct.pack("!I", message_length)
    sock.send(header + message.encode())

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 12345))
server_socket.listen(1)

cap = cv2.VideoCapture(0)
if cap is None or not cap.isOpened():
    print("bad camera")
    exit(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.1)

model_d = pickle.load(open('./model.pickle', 'rb'))
model = model_d['model']

# wait for program to connect
print("waiting for client")
#conn, addr = server_socket.accept()

while True:
    ret, frame = cap.read()

    H, W, _ = frame.shape

    res = hands.process(frame)
    if res.multi_hand_landmarks:
        X = []
        Y = []
        temp = []
        hand_landmarks = res.multi_hand_landmarks[0]
        for i in range(len(hand_landmarks.landmark)):
            X.append(hand_landmarks.landmark[i].x)
            Y.append(hand_landmarks.landmark[i].y)

        x_min = min(X)
        y_min = min(Y)
        x_max = max(X)
        y_max = max(Y)

        x_mid = (x_min + x_max)/2
        y_mid = (y_min + y_max)/2

        for i in range(len(hand_landmarks.landmark)):
            temp.append(X[i] - x_min)
            temp.append(Y[i] - y_min)

        y_pred = model.predict([temp])
        print(y_pred, x_mid, y_mid, type(x_mid), type(y_mid))
        message_action = y_pred[0]
        message_x = round(x_mid, 4)
        message_y = round(y_mid, 4)
        message = f"{message_action},{1-message_x},{message_y}"
        print(message)
        #send_message(conn, message)
    
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                  mp_style.get_default_hand_landmarks_style(),
                                  mp_style.get_default_hand_connections_style())

        cv2.circle(frame, (int(x_mid*W), int(y_mid*H)), 5, (255, 0, 0), 2)
    cv2.circle(frame, (int(W/2), int(H/2)), int(0.16*W), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    cv2.waitKey(6)

cap.release()
cv2.destroyAllWindows()
#conn.close()
#server_socket.close()
