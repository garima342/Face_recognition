# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 20:51:21 2025

@author: hp
"""

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)

    cv2.imshow('Face Detection', img)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:  
        break

cap.release()
cv2.destroyAllWindows()
