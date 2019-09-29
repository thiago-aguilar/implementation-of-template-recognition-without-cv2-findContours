import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('alvo.jpg',0)

cv2.imshow('image',img)
imgplot = plt.imshow(img)


  
  
# capture frames from a camera 
cap = cv2.VideoCapture('entrada.avi') 
  
  
# loop runs if capturing has been initialized 
while(1): 
  
    # reads frames from a camera 
    ret, frame = cap.read() 
  
    # converting BGR to HSV 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    gray = np.float32(gray)  
    # define range of red color in HSV 
    lower_red = np.array([30,150,50]) 
    upper_red = np.array([255,255,180]) 
      
    # create a red HSV colour boundary and  
    # threshold HSV image 
    mask = cv2.inRange(hsv, lower_red, upper_red) 
    dst = cv2.cornerHarris(gray,2,3,0.04)
    # Bitwise-AND mask and original image 
    res = cv2.bitwise_and(frame,frame, mask= mask) 
  
    # Display an original image 
    cv2.imshow('Original',frame) 
  
    # finds edges in the input image image and 
    # marks them in the output map edges 
    edges = cv2.Canny(frame,100,200,300)
    frame[dst>0.01*dst.max()]=[0,0,255] 
  
    # Display edges in a frame 
    cv2.imshow('Edges',edges) 
    cv2.imshow('Corners', frame)	
    # Wait for Esc key to stop 
    k = cv2.waitKey(5) & 0xFF
    if k == 27: 
        break
  
  
### Close the window 
cap.release() 
#  
# De-allocate any associated memory usage 
cv2.destroyAllWindows() 
