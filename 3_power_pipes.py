import cv2
import numpy as np
from scipy.spatial import distance
import pandas as pd


image = cv2.imread('f2.bmp')
cv2.imshow('original', cv2.resize(image, (800, 600)))
# (hMin = 0 , sMin = 0, vMin = 93), (hMax = 179 , sMax = 255, vMax = 174)

# contrast = 2.7
# brightness = -180
# out = cv2.addWeighted( image, contrast, image, 0, brightness)
# cv2.imshow('lightening change', cv2.resize(out, (800, 600)))

#COLORMAP_RAINBOW
#COLORMAP_BONE
#COLORMAP_HSV

# image = cv2.applyColorMap(out, cv2.COLORMAP_RAINBOW)
# cv2.imshow('applyColorMap', cv2.resize(image, (800, 600)))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 50, 255)
cv2.imshow('Canny Edges After Contouring', cv2.resize(edged, (800, 600)))


kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(edged, kernel, iterations=1)
cv2.imshow('dilated image', cv2.resize(img_dilation, (800, 600)))

detected_circles = cv2.HoughCircles(img_dilation,  
                   cv2.HOUGH_GRADIENT, 1, 25, param1 = 10, 
               param2 = 8, minRadius = 2, maxRadius = 5) 
  
# Draw circles that are detected. 
if detected_circles is not None: 
    # Convert the circle parameters a, b and r to integers. 
    detected_circles = np.uint16(np.around(detected_circles)) 
  
    for pt in detected_circles[0, :]: 
        a, b, r = pt[0], pt[1], pt[2] 
  
        # Draw the circumference of the circle. 
        cv2.circle(image, (a, b), r, (0, 255, 0), 2) 
  
        # Draw a small circle (of radius 1) to show the center. 
        cv2.circle(image, (a, b), 1, (0, 0, 255), 3) 
    cv2.imshow("Detected Circle", cv2.resize(image, (800, 600))) 


cv2.waitKey(0)
cv2.destroyAllWindows()