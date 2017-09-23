# import the necessary packages
#from __future__ import print_function
from imutils import paths
from imutils.object_detection import non_max_suppression
import numpy as np
# import argparse
import imutils
import cv2

from PIL import Image
import io
from io import BytesIO 		#Byte stream handler
import os
import tempfile

def detectHuman(path, imgpath):
	print(path, '\n')
	print('\n')
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	cv_image = cv2.imread(path)
	cv_image = imutils.resize(cv_image, width=min(400, cv_image.shape[1]))

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(cv_image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.85)
	 
	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(cv_image, (xA, yA), (xB, yB), (0, 255, 0), 2)
	
	f = os.path.join(imgpath, "detectedimage.jpg")
	cv2.imwrite(f, cv_image)
	cv2.imshow(f, cv_image)
	return f




	