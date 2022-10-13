
from fileinput import filename
import os
import cv2
from cvzone.HandTrackingModule import HandDetector
FJoin = os.path.join

def GetFiles(path):
	file_list, dir_list = [], []
	for dir, subdirs, files in os.walk(path):
		file_list.extend([FJoin(dir, f) for f in files])
		dir_list.extend([FJoin(dir, d) for d in subdirs])
	file_list = filter(lambda x: not os.path.islink(x), file_list)
	dir_list = filter(lambda x: not os.path.islink(x), dir_list)
	return file_list, dir_list


files, dirs = GetFiles(os.path.expanduser("F:\Hand_Pose_3D\images\HOI4D_release\ZY20210800004\H4"))# H3, H4 c2-c9
folder_left=('F:\Hand_Pose_3D\images\handpose_train\Refinehandpose_left')
folder_right=('F:\Hand_Pose_3D\images\handpose_train\Refinehandpose_right')

for file in files:

		#linktrain
		linka=file.strip('\image.mp4')
		link=linka+'\image.mp4'
		cap = cv2.VideoCapture(file)
		detector = HandDetector(detectionCon=0.8, maxHands=2)
		dem=-1
		
		#linksave:
		print(file)
		chiso=file.rfind('\Z') 
		print(file[0:chiso])
		#link = file[0:chiso]
		name_train=file[chiso+1:len(linka)].rstrip('\{}'.format('align_rgb'))
		print(name_train)

