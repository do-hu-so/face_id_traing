from unittest import findTestCases
import cv2
import numpy as np
import face_recognition
import os

path = 'ImagesAttendace'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncoding(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodeListKnown = findEncoding(images)
print('Encoding Comlete')

cap = cv2.VideoCapture(0)
# img = face_recognition.load_image_file('ImagesAttendace/hoang.jpg')
# (h, w, d) = img.shape
# r = 300.0 / w 
# dim = (300, int(h * r))
# resized = cv2.resize(img, dim)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0), None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    

    faceCurFrame = face_recognition.face_locations(imgS)
    
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        n=np.sum(float(faceDis))
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
           # y1,x2,y2,x1 =  y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        if 1.5<n:
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,"dell biet ai",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)      

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)



# faceLoc = face_recognition.face_locations(imgHoang)[0]
# encodeHoang = face_recognition.face_encodings(imgHoang)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(255,0,255),2)

# facetest = face_recognition.face_locations(imgTest)[0]
# encodeHoangtest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgElontest,(facetest[3],facetest[0],facetest[1],facetest[2]),(255,0,255),2)

# results = face_recognition.compare_faces([encodeHoang],encodeHoangtest)
# faceDis = face_recognition.face_distance([encodeHoang],encodeHoangtest)