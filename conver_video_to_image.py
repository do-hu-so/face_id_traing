# Importing all necessary libraries
import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import face_recognition
import numpy as np
# Read the video from specified path
cap = cv2.VideoCapture(linhtest.py)
data = np.loadtxt("Data.csv")

try:

    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')

# if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0
detector = HandDetector(detectionCon=0.8, maxHands=2)
while (True):

    # reading from frame
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0), None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_RGB2BGR)
    
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    if success:
        for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
            matches = face_recognition.compare_faces(data,encodeFace)
            faceDis = face_recognition.face_distance(data,encodeFace)
            #print(faceDis)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                # continue creating images until video remains
                name = '/mnt/backup1/SON/WorkSpace/ImagesAttendace/PHP_Linh/A_image/{}'.format(str(currentframe)) + '.jpg'
                print('Creating...' + name)

                # writing the extracted images
                cv2.imwrite(name, img)

                # increasing counter so that it will
                # show how many frames are created
                
            else:
                name = 'F:\Hand_Pose_3D\{}env\data\Khong_Phat_Hien'.format('v') + '\{}'.format(str(currentframe)) + '.jpg'
                print('Creating...' + name)

                # writing the extracted images
                cv2.imwrite(name, img)

                # increasing counter so that it will
                # show how many frames are created
            
        currentframe += 1
    else:
        break

# Release all space and windows once done
cap.release()
cv2.destroyAllWindows()

