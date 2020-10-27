import cv2
import numpy as np
from scipy.spatial import distance
import pandas as pd
import os

#(hMin = 5 , sMin = 0, vMin = 0), (hMax = 120 , sMax = 255, vMax = 255)
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

def filter_distances(lines, thresh = 5):
    cordinates = lines # sort_lines(lines)

    new_cor = []
    for i in range(len(cordinates)):
        cur = i
        nex = i + 1 if i != (len(cordinates) -1) else cur
        pre = i - 1 if i != 0 else cur

        if (np.abs(cordinates[cur][1] - cordinates[cur][3]) < 10) or (np.abs(cordinates[cur][0] - cordinates[cur][2]) < 2):
            continue

        nex_dist = distance.euclidean(cordinates[cur], cordinates[nex])
        prev_dist = distance.euclidean(cordinates[pre], cordinates[cur])

        if (nex_dist >= thresh and prev_dist >= thresh) or nex == cur or pre == cur:
            new_cor.append(cordinates[cur])
            # print('The coordinates -> ', cordinates[cur], cordinates[nex])
            # print('The Dist -> ', nex_dist)
    return new_cor

def get_slop_intercept(points):
    x1, y1, x2, y2 = points
    slope = (y2 - y1) // (x2 - x1) if (x2 - x1) != 0 else None
    if slope == None:
        return None, None
    intercept = y1 - (slope * x1)
    return int(slope) , int(intercept)

def get_points_on_line(points):
    slope, intercept = get_slop_intercept(points)
    if slope == None:
        return None, None
    xs = np.array(range(1024))
    ys = xs * slope + intercept
    return xs.astype(int), ys.astype(int)

def get_all_slopes(lines):
    data_array = []
    lines = sort_lines(lines)
    for line in lines:
        data_array.append(get_slop_intercept(line))
    return data_array

def filter_with_slopes(lines, ts, ti):
    result_array = []
    data_array = get_all_slopes(lines)
    slope = 0
    intercept = 0
    info_array = []
    for i in range(len(data_array)-1):
        next_slope = data_array[i + 1][0] if (i + 1 != len(data_array)) and (data_array[i + 1][0] != None) else slope
        next_intercept = data_array[i + 1][1] if (i + 1 != len(data_array)) and (data_array[i + 1][1] != None) else intercept
        
        slope = data_array[i][0] if data_array[i][0] != None else 0
        intercept = data_array[i][1] if data_array[i][1] != None else 0
        # print(slope, next_slope)
        
        if next_slope == 0 or next_intercept == 0:
            result_array.append(False)
            continue
        
        slope_ratio = abs(slope/next_slope)
        intercept_ratio = abs(intercept/next_intercept)
        # print('[!!!]', slope_ratio, intercept_ratio)
        info_array.append((slope_ratio, intercept_ratio))

        if 1.05 > slope_ratio > ts and 1.05 > intercept_ratio > ti:
            result_array.append(False)
        else:
            result_array.append(True)
    result_array.append(True)
    info_array.append('Last element')

    return lines[result_array], info_array

def m_filter_with_slopes(lines, ts_low, ts_high, ti_low, ti_high):
    result_array = []
    another_array = []
    lines = sort_lines(lines)
    for line in lines:

        # the lines on the edges are not relevant to us
        if (np.abs(line[1] - line[3]) < 10) or (np.abs(line[0] - line[2]) < 2):
                continue

        slope, intercept = get_slop_intercept(line)
        slope = 0 if slope is None else slope
        intercept = 0 if intercept is None else intercept

        # first member of the array must be added manually
        if len(result_array) == 0:
            result_array.append(line)
            # continue

        for l2 in result_array:
            l2_slope, l2_intercept = get_slop_intercept(l2)     
            l2_slope = 0 if l2_slope is None else l2_slope
            l2_intercept = 0 if l2_intercept is None else l2_intercept

            slope_ratio = abs(l2_slope/slope) if slope != 0 else 0
            intercept_ratio = abs(l2_intercept/intercept) if intercept != 0 else 0
            
            # print('[*_*] Info :', slope_ratio, intercept_ratio)
            # print('[+] current:', slope, intercept)
            # print('[-] comp:', l2_slope, l2_intercept)
            # print('[++] lines :', line, l2)
            if (slope_ratio < ts_low or slope_ratio > ts_high) and (intercept_ratio < ti_low or intercept_ratio > ti_high):
                another_array.append(True)
            else:
                another_array.append(False)
            # print('====================================================================')
        
        if np.all(another_array):
            print('[!] Rigistered!', line)
            result_array.append(line)
        another_array = []

    return result_array

results_path = os.path.join(os.getcwd(), 'results')
test_path = os.path.join(os.getcwd(), 'test')

image = cv2.imread(os.path.join(test_path, 'a2.bmp'))
cv2.imshow('original', cv2.resize(image, (800, 600)))
# cv2.imwrite(os.path.join(results_path, 'original.jpg'), cv2.resize(image, (800, 600)))
# cv2.waitKey(0)

# lineImage = image.copy()
# clineImage = image.copy()

# contrast = 2.5
# brightness = -240
# out = cv2.addWeighted( image, contrast, image, 0, brightness)
# cv2.imshow('lightening change', cv2.resize(out, (800, 600)))
# cv2.imwrite('lightening change.jpg', cv2.resize(out, (800, 600)))

# cv2.waitKey(0)
# image = out.copy()
#COLORMAP_RAINBOW
#COLORMAP_BONE
#COLORMAP_HSV

# c1 = cv2.applyColorMap(image, cv2.COLORMAP_HSV)
# cv2.imshow('applyColorMap1', cv2.resize(c1, (800, 600)))

# c2 = cv2.applyColorMap(image, cv2.COLORMAP_RAINBOW)
# cv2.imshow('applyColorMap2', cv2.resize(c2, (800, 600)))

# c3 = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
# cv2.imshow('applyColorMap3', cv2.resize(c3, (800, 600)))

# c4 = cv2.applyColorMap(image, cv2.COLORMAP_DEEPGREEN)
# cv2.imshow('applyColorMap4', cv2.resize(c4, (800, 600)))

# c15 = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
# cv2.imshow('applyColorMap5', cv2.resize(c15, (800, 600)))

c5 = cv2.applyColorMap(image, cv2.COLORMAP_TWILIGHT_SHIFTED)
cv2.imshow('applyColorMap6', cv2.resize(c5, (800, 600)))
# (hMin = 0 , sMin = 0, vMin = 0), (hMax = 179 , sMax = 130, vMax = 255)

c6 = cv2.applyColorMap(image, cv2.COLORMAP_TWILIGHT)
cv2.imshow('applyColorMap7', cv2.resize(c6, (800, 600)))

hsv = cv2.cvtColor(c5, cv2.COLOR_BGR2HSV)

# Threshold of blue in HSV space
lower_blue = np.array([0, 0, 0])
upper_blue = np.array([179, 130, 255])

# preparing the mask to overlay
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# The black region in the mask has the value of 0,
# so when multiplied with original image removes all non-blue regions
result = cv2.bitwise_and(c5, c5, mask = mask)

cv2.imshow('pipe color filtered', cv2.resize(result, (800, 600)))

# c7 = cv2.applyColorMap(image, cv2.COLORMAP_CIVIDIS)
# cv2.imshow('applyColorMap8', cv2.resize(c7, (800, 600)))

# c8 = cv2.applyColorMap(image, cv2.COLORMAP_VIRIDIS)
# cv2.imshow('applyColorMap9', cv2.resize(c8, (800, 600)))

# c9 = cv2.applyColorMap(image, cv2.COLORMAP_PLASMA)
# cv2.imshow('applyColorMap10', cv2.resize(c9, (800, 600)))

# c10 = cv2.applyColorMap(image, cv2.COLORMAP_INFERNO)
# cv2.imshow('applyColorMap11', cv2.resize(c10, (800, 600)))

# c11 = cv2.applyColorMap(image, cv2.COLORMAP_MAGMA)
# cv2.imshow('applyColorMap12', cv2.resize(c11, (800, 600)))

# c12 = cv2.applyColorMap(image, cv2.COLORMAP_PARULA)
# cv2.imshow('applyColorMap13', cv2.resize(c12, (800, 600)))

# c13 = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
# cv2.imshow('applyColorMap14', cv2.resize(c13, (800, 600)))

# c14 = cv2.applyColorMap(image, cv2.COLORMAP_PINK)
# cv2.imshow('applyColorMap15', cv2.resize(c14, (800, 600)))



cv2.waitKey(0)
quit()

# Find Canny edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 50, 255)
# cv2.waitKey(0)

cv2.imshow('Canny Edges After Contouring', cv2.resize(edged, (800, 600)))
cv2.imwrite(os.path.join(results_path, 'Canny Edges After Contouring.jpg'), cv2.resize(edged, (800, 600)))

# kernel = np.ones((7,1), np.uint8)
# img_eroded = cv2.erode(edged, kernel, iterations=1)
# cv2.imshow('eroded image', cv2.resize(img_eroded, (800, 600)))

kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(edged, kernel, iterations=1)
cv2.imshow('dilated image', cv2.resize(img_dilation, (800, 600)))
cv2.imwrite(os.path.join(results_path, 'dilated image.jpg'), cv2.resize(img_dilation, (800, 600)))
# cv2.waitKey(0)
# quit()
# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contours, hierarchy = cv2.findContours(img_dilation,
    cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print("Number of Contours found = " + str(len(contours)))

# cv2.waitKey(0)
# quit()

inverted = cv2.bitwise_not(edged)
cv2.imshow('Inverted', cv2.resize(inverted, (800, 600)))
cv2.imwrite(os.path.join(results_path, 'Inverted.jpg'), cv2.resize(inverted, (800, 600)))
# cv2.waitKey(0)


# Draw all contours
# -1 signifies drawing all contours
# print(contours)
# new_contours = []
# high = []
# width = []
# area = []
# for c in contours:
#     (x, y, w, h) = cv2.boundingRect(c)
#     ratio = h/w
#     high.append(h)
#     width.append(w)
#     a = cv2.contourArea(c)
#     #0.0 467.0 16.797619047619047
#     area.append(a)
#     if 5000 < a < 950000:
#         new_contours.append(c)
#         # print(c)

# high = np.array(high)
# width = np.array(width)
# area = np.array(area)
# new_contours = np.array(new_contours)

# print('[+] height:', high.min(), high.max(), high.mean())
# print('[-] width:', width.min(), width.max(), width.mean())
# print('[!] area:', area.min(), area.max(), area.mean())


cv2.drawContours(image, contours, -1, (0, 0, 255))#, thickness = cv2.FILLED)
cv2.imshow('Contours', cv2.resize(image, (800, 600)))
cv2.imwrite(os.path.join(results_path, 'Contours.jpg'), cv2.resize(image, (800, 600)))
# cv2.waitKey(0)

lines = cv2.HoughLinesP(img_dilation, 1, np.pi/500, 50, np.array([]), 400, 20)
print('[++]', np.array(lines).shape)
# cordinates = []
# if lines is not None:
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             # slope, intercept, r_value, p_value, std_err = linregress([x1,x2],[y1,y2])
#             # x = np.linspace(0, 100, 100)
#             # y = slope * x + intercept
#             # cordinates.append((x1, y1, x2, y2))
#             cv2.line(lineImage, (x1, y1), (x2, y2), (20, 220, 20), 2)

# distances = []
# for i in range(len(cordinates[1:])):
#     dist = distance.euclidean(cordinates[i-1], cordinates[i])
#     distances.append(dist)

# distances = np.array(distances)
# print('[--]', distances.min(), distances.max(), distances.mean())
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


# polys = np.array_split(new_lines, len(new_lines) // 2)
# for poly in polys:
#     pts = np.array_split(poly.ravel(), len(poly.ravel()) // 2)
#     c = 0
#     for i in range(len(pts) - 1):
#         c += 1
#         if c % 3 == 0:
#             t = pts[i]
#             pts[i] = pts[i + 1]
#             pts[i + 1] = t
#     color = (255, 0, 0)
#     cv2.fillPoly(lineImage, [np.array(pts)],color)

# cv2.imshow('pool', cv2.resize(lineImage, (800, 600)))

cv2.waitKey(0)
cv2.destroyAllWindows()

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # It converts the BGR color space of image to HSV color space
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Threshold of blue in HSV space
# lower_blue = np.array([0, 0, 40])
# upper_blue = np.array([179, 255, 207])

# # preparing the mask to overlay
# mask = cv2.inRange(hsv, lower_blue, upper_blue)

# # The black region in the mask has the value of 0,
# # so when multiplied with original image removes all non-blue regions
# result = cv2.bitwise_and(image, image, mask = mask)

# cv2.imshow('pipe color filtered', cv2.resize(result, (800, 600)))

# # result = image
# gray = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)
# gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)

# thresh = 80
# im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow('binary', cv2.resize(im_bw, (800, 600)))


# # cv2.imshow('res', cv2.resize(result, (800, 600)))

# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
