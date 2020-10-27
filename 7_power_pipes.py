import cv2
import numpy as np
import os
from pp_utils import wall_thickness_detector


results_path = os.path.join(os.getcwd(), 'results')
test_path = os.path.join(os.getcwd(), 'test')

wtd = wall_thickness_detector(test_path, results_path)

image = cv2.imread(os.path.join(test_path, 'a2.bmp'))

full_image = cv2.imread(os.path.join(test_path, '870-871-872-874_002 (2).bmp'))

cv2.imshow('original', cv2.resize(image, (800, 600)))
# cv2.imwrite(os.path.join(results_path, 'original.jpg'), cv2.resize(image, (800, 600)))

# de_image = cv2.fastNlMeansDenoisingColored(image, None, 9, 9, 7, 21)
# cv2.imshow('Fast Denoising', cv2.resize(de_image, (800, 600)))

de_image = cv2.bilateralFilter(image, 9, 75, 75)
# cv2.imshow('Bilateral', cv2.resize(de_image, (800, 600)))
  
lineImage = image.copy()
clineImage = image.copy()
result_image = image.copy()

gray = cv2.cvtColor(de_image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 50, 255)
# cv2.imshow('Canny Edges After Contouring', cv2.resize(edged, (800, 600)))
# cv2.imwrite(os.path.join(results_path, 'Canny Edges After Contouring.jpg'), cv2.resize(edged, (800, 600)))

kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(edged, kernel, iterations=1)
# cv2.imshow('dilated image', cv2.resize(img_dilation, (800, 600)))

# kernel = np.ones((5,5), np.uint8)
# img_dilation = cv2.erode(img_dilation, kernel, iterations=1)
# cv2.imshow('eroded image', cv2.resize(img_dilation, (800, 600)))
# cv2.imwrite(os.path.join(results_path, 'dilated image.jpg'), cv2.resize(img_dilation, (800, 600)))
# cv2.waitKey(0)
# quit()
contours, hierarchy = cv2.findContours(img_dilation,
    cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# print("Number of Contours found = " + str(len(contours)))

cv2.drawContours(image, contours, -1, (0, 0, 255))#, thickness = cv2.FILLED)
# cv2.imshow('Contours', cv2.resize(image, (800, 600)))
# cv2.imwrite(os.path.join(results_path, 'Contours.jpg'), cv2.resize(image, (800, 600)))
""" ============================= Here is the HoughLinesP Thresh ==================================== """
lines = cv2.HoughLinesP(img_dilation, 1, np.pi/500, 50, np.array([]), 300, 0)
print('[++]', np.array(lines).shape)

new_lines = wtd.sort_lines(lines)
for x1, y1, x2, y2  in new_lines:
    xs, ys = wtd.get_points_on_line((x1, y1, x2, y2))
    if xs is not None:
        cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for x, y in zip(xs, ys):
            cv2.circle(lineImage, (x,y), 2, (0,0,255), 2)
# cv2.imshow('res', cv2.resize(lineImage, (800, 600)))
# cv2.imwrite(os.path.join(results_path, 'res.jpg'), cv2.resize(lineImage, (800, 600)))
# cv2.waitKey(0)
# quit()
# filtered_lines = m_filter_with_slopes(lines, 0.99, 1.05, 0.99, 1.05)
filtered_lines = wtd.filter_with_point(lines, 512, 20)
print('[--]', len(filtered_lines))

for x1, y1, x2, y2  in filtered_lines:
    xs, ys = wtd.get_points_on_line((x1, y1, x2, y2))
    cv2.line(clineImage, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if xs is not None:
        for x, y in zip(xs, ys):
            cv2.circle(clineImage, (x,y), 2, (255,0,0), 2)
# cv2.imshow('res filtered', cv2.resize(clineImage, (800, 600)))
# cv2.waitKey(0)
# quit()
error_list = []
ct, ce = wtd.check_thickness(filtered_lines, 4, 3)
print(ct, ce)

wtd.draw_flaws(result_image, filtered_lines, ct, ce)

error_list = [i + 1 for i, c in enumerate(ct) if c]
# for i, c in enumerate(ct):
#     if c:
#         error_list.append(i + 1)

if len(error_list):
    cv2.putText(clineImage, f'there is a wall thickness error in tupe number {error_list}', (10,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2)
    cv2.putText(result_image, f'there is a wall thickness error in tupe number {error_list}', (10,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2)
if True in ce:
    edges = [i + 1 for i, e in enumerate(ce) if e]
    cv2.putText(clineImage, f'there is an edge thickness error on edges {edges}', (10,90), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2)
    cv2.putText(result_image, f'there is an edge thickness error on edges {edges}', (10,90), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2)
else:
    cv2.putText(clineImage, 'No wall thickness error detected', (10,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2)
    cv2.putText(result_image, 'No wall thickness error detected', (10,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2)

cv2.imshow('res filtered', cv2.resize(clineImage, (800, 600)))
cv2.imshow('Final Result', cv2.resize(result_image, (800, 600)))
cv2.imwrite(os.path.join(results_path, 'wall_thickness 2.jpg'), cv2.resize(clineImage, (800, 600)))
cv2.imwrite(os.path.join(results_path, 'final result.jpg'), cv2.resize(result_image, (800, 600)))

cv2.waitKey(0)
cv2.destroyAllWindows()