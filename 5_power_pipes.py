import cv2
import numpy as np
import os

def slope_intercept(x1,y1,x2,y2):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a,b

def sort_lines(lines):
    cordinates = []
    xs = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                xs.append(x1)
                cordinates.append((x1, y1, x2, y2))
    xs = np.array(xs)
    cordinates = np.array(cordinates)[xs.argsort()]
    return cordinates.tolist()

def get_slop_intercept(points):
    x1, y1, x2, y2 = points
    slope = (y2 - y1) // (x2 - x1) if (x2 - x1) != 0 else None
    if slope == None:
        return None, None
    intercept = y1 - (slope * x1)
    return int(slope) , int(intercept)

results_path = os.path.join(os.getcwd(), 'results')
test_path = os.path.join(os.getcwd(), 'test')

image = cv2.imread(os.path.join(test_path, 'a2.bmp'))
cv2.imshow('original', cv2.resize(image, (800, 600)))
cv2.imwrite(os.path.join(results_path, 'original.jpg'), cv2.resize(image, (800, 600)))

lineImage = image.copy()
clineImage = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 50, 255)
# cv2.imshow('Canny Edges After Contouring', cv2.resize(edged, (800, 600)))
# cv2.imwrite(os.path.join(results_path, 'Canny Edges After Contouring.jpg'), cv2.resize(edged, (800, 600)))

kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(edged, kernel, iterations=1)
# cv2.imshow('dilated image', cv2.resize(img_dilation, (800, 600)))
# cv2.imwrite(os.path.join(results_path, 'dilated image.jpg'), cv2.resize(img_dilation, (800, 600)))

contours, hierarchy = cv2.findContours(img_dilation,
    cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# print("Number of Contours found = " + str(len(contours)))

cv2.drawContours(image, contours, -1, (0, 0, 255))#, thickness = cv2.FILLED)
# cv2.imshow('Contours', cv2.resize(image, (800, 600)))
# cv2.imwrite(os.path.join(results_path, 'Contours.jpg'), cv2.resize(image, (800, 600)))

lines = cv2.HoughLinesP(img_dilation, 1, np.pi/180, 50, np.array([]), 400, 20)
print('[++]', np.array(lines).shape)

circle_lines = []
new_lines = sort_lines(lines)
for x1, y1, x2, y2  in new_lines:
    xs, ys = get_points_on_line((x1, y1, x2, y2))
    if xs is not None:
        # cv2.line(lineImage, (xs[0], ys[0]), (xs[-1], ys[-1]), (255, 0, 0), 2)
        cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 0), 2)
        circle_lines.append((xs[0], ys[0], xs[-1], ys[-1]))
        for x, y in zip(xs, ys):
            cv2.circle(lineImage, (x,y), 2, (0,0,255), 2)
cv2.imshow('res', cv2.resize(lineImage, (800, 600)))
cv2.imwrite(os.path.join(results_path, 'res.jpg'), cv2.resize(lineImage, (800, 600)))

filtered_lines = m_filter_with_slopes(lines, 0.99, 1.05, 0.99, 1.05)
print('[--]', len(filtered_lines))

for x1, y1, x2, y2  in filtered_lines:
    cv2.line(clineImage, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('res filtered', cv2.resize(clineImage, (800, 600)))
cv2.imwrite(os.path.join(results_path, 'res filtered.jpg'), cv2.resize(clineImage, (800, 600)))


cv2.waitKey(0)
cv2.destroyAllWindows()