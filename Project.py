
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

data = np.loadtxt("Data.csv")
FJoin = os.path.join
path = 'D:\Workspace\{}CKH_Th_Nhat\ImagesAttendace'.format('N')
images = []
classNames = []

def GetFiles(path):
	file_list, dir_list = [], []
	for dir, subdirs, files in os.walk(path):
		file_list.extend([FJoin(dir, f) for f in files])
		dir_list.extend([FJoin(dir, d) for d in subdirs])
	file_list = filter(lambda x: not os.path.islink(x), file_list)
	dir_list = filter(lambda x: not os.path.islink(x), dir_list)
	return file_list, dir_list
files, dirs = GetFiles(os.path.expanduser(path))#path file
#load name
for file in files:
    #print(file)
    chiso=file.rfind('\A_image') 
    curImg = cv2.imread(file)

    a=file[0:chiso]
    #print(a)
    b=file[0:chiso].lstrip('\{}'.format('D:\Workspace\{}CKH_Th_Nhat'.format('N')))
    name_train=b.lstrip('ImagesAttendace{}'.format('\o'))
    #print(name_train)

    images.append(curImg)
    classNames.append(os.path.splitext(name_train)[0])
    #print('===================')
print(classNames)


with open('name.txt', 'w') as wf:
    for text in classNames:
        for imgdata in text:
            wf.writelines(text + '\n')
            break
def markAttendace(name):
    with open('Attendace.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

img = face_recognition.load_image_file('image_train/Hoang_Nam.jpg')
#cap = cv2.VideoCapture('video3.mp4')


while True:
#    success, img = cap.read()
    imgS = cv2.resize(img,(0,0), None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
        
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(data,encodeFace)
        faceDis = face_recognition.face_distance(data,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
        #if faceDis < 0.40:
        if matches[matchIndex]:
              name = classNames[matchIndex].upper()
              print('phat hien: '+name)
              #print('day la matches',matches)
              y1,x2,y2,x1 = faceLoc
              y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
              cv2.rectangle(img,(x1,y1),(x2,y2) ,(255,0,255),2)
              cv2.rectangle(img, (x1,y2 - 35),(x2,y2),(0,255,0),2)
              cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
              markAttendace(name)
              print('===========')
    imgS = cv2.resize(imgS, (0, 0), None, 1, 1) 
    cv2.imshow('Webcam',imgS)
    cv2.waitKey(1)