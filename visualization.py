from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import os

path=os.path.join(os.path.abspath(os.curdir) , 'my_model.onnx')
args_confidence = 0.8
CLASSES = ['face', 'not face']
print("[INFO] loading model...")
net = cv2.dnn.readNetFromONNX (path)
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
frame = vs.read()
frame = imutils.resize(frame, width=400)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (32, 32)),scalefactor=1.0/32
                              , size=(32, 32), mean= (128,128,128), swapRB=True)
	cv2.imshow("Cropped image", cv2.resize(frame, (32, 32)))
	net.setInput(blob)
	detections = net.forward()
	print(list(zip(CLASSES,detections[0])))
	confidence = abs(detections[0][0]-detections[0][1])
	print("confidence = ", confidence)
	if (confidence > args_confidence) :
		class_mark=np.argmax(detections)
		cv2.putText(frame, CLASSES[class_mark], (30,30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (242, 230, 220), 2)
	else:
		class_mark=np.argmax(detections)
		cv2.putText(frame,None,(30,30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (242, 230, 220), 2)

	cv2.imshow("Web camera view", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

fps.stop()
cv2.destroyAllWindows()
vs.stop()
