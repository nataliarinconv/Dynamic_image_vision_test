import numpy as np
import cv2

#inside the parenthesis is just where you want to get
#the video from

# name = 'output' + '.mp4'
cap = cv2.VideoCapture('MinionsDance.mp4')

#Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC(*'MP4V')
output = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640,480))

while(cap.isOpened()):
	#Capture frame-by-frame
	#cap.read() returns a bool depending on whether or not 
	#the frame was correctly read
	ret, frame = cap.read()
	if ret==True:
    #     frame = cv2.flip(frame, 0)
    
	    #Our operations on the frame come here (turn to gray)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	    #write flipped and gray frame
		output.write(gray)

		#blurred = cv2.GaussianBlur(gray, (9,9), 0)
		#edgeDetected = cv2.Canny(blurred, 50, 150)
		#Display the resulting frame
		cv2.imshow('frame', gray)

		if cv2.waitKey(1) & 0xFF == ord('q'):
		    break
	else:
	    break

#When everythings done, release the capture
cap.release()
output.release()
cv2.destroyAllWindows()


# You can also access some of the features of this 
# video using cap.get(propId) method where propId 
# is a number from 0 to 18. Each number denotes a 
# property of the video (if it is applicable to that 
# video) and full details can be seen here: Property 
# Identifier. Some of these values can be modified 
# using cap.set(propId, value). Value is the new 
# value you want.


