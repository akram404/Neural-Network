# this creates box around deteced face with RGB colors changing

import cv2
import os
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
a,b,c,d = 0,0,0,1
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (a,b,c), d)
        
    if(d==10):
        d = 5
    else:
        d=10
    c = c + 50
    if(c == 250):
        c = 0
        b = b + 50
    if(b == 250):
        b = 0
        a = a + 50
    if(a == 250):
        a = 0
    cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
