import numpy as np
import cv2

#inside the parenthesis is just where you want to get
#the video from
cap = cv2.VideoCapture('MinionsDance.mp4')
#the frame was correctly read
ret, frame = cap.read()

while(cap.isOpened()):
	#Capture frame-by-frame
	#cap.read() returns a bool depending on whether or not 
	#the frame was correctly read
	#ret, frame = cap.read()
        
        if ret == False:
           break

	#Our operations on the frame come here (turn to gray)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	blurred = cv2.GaussianBlur(gray, (9,9), 0)

	edgeDetected = cv2.Canny(blurred, 50, 150)
	#Display the resulting frame
	cv2.imshow('frame', edgeDetected)
	#the frame was correctly read
	ret, frame = cap.read()
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

#When everythings done, release the capture
cap.release()
cv2.destroyAllWindows()


# You can also access some of the features of this 
# video using cap.get(propId) method where propId 
# is a number from 0 to 18. Each number denotes a 
# property of the video (if it is applicable to that 
# video) and full details can be seen here: Property 
# Identifier. Some of these values can be modified 
# using cap.set(propId, value). Value is the new 
# value you want.

#good website to think about image comparison
#https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/

