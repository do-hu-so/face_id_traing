import cv2
import numpy as np
import face_recognition
import os
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
    print(file)
    chiso=file.rfind('\A_image') 
    curImg = cv2.imread(file)

    a=file[0:chiso]
    print(a)
    b=file[0:chiso].lstrip('\{}'.format('D:\Workspace\{}CKH_Th_Nhat'.format('N')))
    name_train=b.lstrip('ImagesAttendace{}'.format('\o'))
    print(name_train)

    images.append(curImg)
    classNames.append(os.path.splitext(name_train)[0])
    print('===================')


print(classNames)

def findEncoding(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodeListKnown = findEncoding(images)
print('Encoding Comlete')
print(encodeListKnown)
np.savetxt('Data.csv', encodeListKnown)

print("da in xong")

            
