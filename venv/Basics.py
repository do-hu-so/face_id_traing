from base64 import encode
from ctypes import resize
from turtle import width
import cv2
import numpy as np
import face_recognition

scale = 60
# hàm xử lý hình ảnh đầu vào
imgHoang = face_recognition.load_image_file('ImagesAttendace/hoang.jpg')
imgHoang = cv2.cvtColor(imgHoang,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('ImagesAttendace/mark.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)


faceLoc = face_recognition.face_locations(imgHoang)[0]
encodeHoang = face_recognition.face_encodings(imgHoang)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(255,0,255),2)

facetest = face_recognition.face_locations(imgTest)[0]
encodeHoangtest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgElontest,(facetest[3],facetest[0],facetest[1],facetest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeHoang],encodeHoangtest)
faceDis = face_recognition.face_distance([encodeHoang],encodeHoangtest)

# hàm chỉnh kích thước
(h, w, d) = imgTest.shape
r = 600.0 / w 
dim = (600, int(h * r))
resized = cv2.resize(imgTest, dim)

# hàm kiểm tra giá trị
if faceDis < 0.50:  
    cv2.putText(resized,"day la hoang",(10,35),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    #cv2.imshow('Elon_Musk',imgElon)
    cv2.imshow('Elon_test',resized)
else:
    cv2.putText(resized,"day khong la hoang",(10,35),cv2.FONT_ITALIC,1,(0,0,255),2)
    #cv2.imshow('Elon_Musk',imgElon)
    cv2.imshow('Elon_test',resized)
cv2.waitKey(0)