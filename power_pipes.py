import cv2
import numpy as np
from scipy.spatial import distance

def filter_lines(lines):
    new_lines = []
    for i in range(len(lines[::2])):
        dst = distance.euclidean(lines[i-1], lines[i])
        # if dst > 500:
        #     new_lines.append(dst)
    
    
    return np.array(new_lines)


image = cv2.imread('737-738-739-740_001.bmp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

# It converts the BGR color space of image to HSV color space 
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    
# Threshold of blue in HSV space 
lower_blue = np.array([0, 0, 100]) 
upper_blue = np.array([179, 255, 219]) 

# preparing the mask to overlay 
mask = cv2.inRange(hsv, lower_blue, upper_blue) 
    
# The black region in the mask has the value of 0, 
# so when multiplied with original image removes all non-blue regions 
result = cv2.bitwise_and(image, image, mask = mask) 

cv2.imshow('pipe color filtered', cv2.resize(result, (800, 600)))

# result = image
gray = cv2.cvtColor(result,cv2.COLOR_HSV2BGR) 
gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY) 

thresh = 60
im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
cv2.imshow('binary', cv2.resize(im_bw, (800, 600)))

# Apply edge detection method on the image 
edges = cv2.Canny(im_bw,150,255) 

  
# This returns an array of r and theta values 
lines = cv2.HoughLinesP(edges, 1, np.pi/500, 50, np.array([]), 100, 5)
# print(len(np.array(lines)[::2]))

if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (20, 220, 20), 3) 

cv2.imshow('res', cv2.resize(image, (800, 600)))

if cv2.waitKey(0) & 0xff == 27: 
    cv2.destroyAllWindows() 