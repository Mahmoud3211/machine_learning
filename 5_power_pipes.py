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

def get_points_on_line(points):
    slope, intercept = get_slop_intercept(points)
    if slope == None:
        return None, None
    xs = np.array(range(1024))
    ys = xs * slope + intercept
    return xs.astype(int), ys.astype(int)

def get_slop_intercept(points):
    x1, y1, x2, y2 = points
    slope = (y2 - y1) // (x2 - x1) if (x2 - x1) != 0 else None
    if slope == None:
        return None, None
    intercept = y1 - (slope * x1)
    return int(slope) , int(intercept)

def filter_irrelevant_lines(lines):
    new_lines = []
    for line in lines:
        if (np.abs(line[1] - line[3]) < 10) or (np.abs(line[0] - line[2]) < 2):
            continue
        else:
            new_lines.append(line)
    return new_lines

def filter_with_point(lines, y=200, tx=10):
    selected_y = y
    lines = filter_irrelevant_lines(sort_lines(lines))
    x = 0
    info_array = []
    for line in lines:
        
        if (np.abs(line[1] - line[3]) < 10) or (np.abs(line[0] - line[2]) < 2):
            info_array.append(0)
            continue

        slope, intercept = get_slop_intercept(line)
        slope = 0 if slope is None else slope
        intercept = 0 if intercept is None else intercept
        x = (selected_y - intercept) // slope if slope != 0 else 0
        info_array.append(x)

    info_array = np.array(info_array)[np.argsort(info_array)]
    lines = np.array(lines)[np.argsort(info_array)]
    new_array = []
    index_array = []
    for i, x in enumerate(info_array):
        if not new_array or abs(x - new_array[-1]) > tx:
            new_array.append(x)
            index_array.append(i)

    return np.array(lines)[index_array]
        
def check_thickness(full_lines, diff_thresh=3, edge_thresh=3):
    y1 = 300
    y2 = 500
    y3 = 700
    y4 = 900

    selected_coor = []
    check_array = []
    edges_array = []
    for group in np.array_split(full_lines, len(full_lines) // 4):
        
        for line in group:
            slope, intercept = get_slop_intercept(line)
            slope = 0 if slope is None else slope
            intercept = 0 if intercept is None else intercept
            
            x1 = (y1 - intercept) // slope if slope != 0 else 0
            x2 = (y2 - intercept) // slope if slope != 0 else 0
            x3 = (y3 - intercept) // slope if slope != 0 else 0
            x4 = (y4 - intercept) // slope if slope != 0 else 0

            selected_coor.append(np.array([x1, x2, x3, x4]))
        dist1 = np.abs(selected_coor[0] - selected_coor[1])
        dist2 = np.abs(selected_coor[2] - selected_coor[3])
        print(dist1, dist2)

        edge_diff = np.all(np.abs(dist1 - dist2) >= edge_thresh)
        right_edge = np.all(dist1.max() - dist1.min() >= diff_thresh)
        left_edge = np.all(dist2.max() - dist2.min() >= diff_thresh)

        edges_array.append(right_edge)
        edges_array.append(left_edge)
        check_array.append(edge_diff)

    return check_array, edges_array



results_path = os.path.join(os.getcwd(), 'results')
test_path = os.path.join(os.getcwd(), 'test')

image = cv2.imread(os.path.join(test_path, 'wt2.bmp'))
cv2.imshow('original', cv2.resize(image, (800, 600)))
# cv2.imwrite(os.path.join(results_path, 'original.jpg'), cv2.resize(image, (800, 600)))

lineImage = image.copy()
clineImage = image.copy()
result_image = image.copy()

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

lines = cv2.HoughLinesP(img_dilation, 1, np.pi/500, 50, np.array([]), 400, 20)
print('[++]', np.array(lines).shape)

new_lines = sort_lines(lines)
for x1, y1, x2, y2  in new_lines:
    xs, ys = get_points_on_line((x1, y1, x2, y2))
    if xs is not None:
        # cv2.line(lineImage, (xs[0], ys[0]), (xs[-1], ys[-1]), (255, 0, 0), 2)
        cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for x, y in zip(xs, ys):
            cv2.circle(lineImage, (x,y), 2, (0,0,255), 2)
cv2.imshow('res', cv2.resize(lineImage, (800, 600)))
# cv2.imwrite(os.path.join(results_path, 'res.jpg'), cv2.resize(lineImage, (800, 600)))

# filtered_lines = m_filter_with_slopes(lines, 0.99, 1.05, 0.99, 1.05)
filtered_lines = filter_with_point(lines, 512, 20)
print('[--]', len(filtered_lines))

for x1, y1, x2, y2  in filtered_lines:
    xs, ys = get_points_on_line((x1, y1, x2, y2))
    cv2.line(clineImage, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if xs is not None:
        for x, y in zip(xs, ys):
            cv2.circle(clineImage, (x,y), 2, (255,0,0), 2)

error_list = []
ct, ce = check_thickness(filtered_lines, 1, 3)
print(ct, ce)
for i, c in enumerate(ct):
    if c:
        error_list.append(i + 1)

if len(error_list):
    cv2.putText(clineImage, f'there is a wall thickness error in tupe number {error_list}', (10,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2)
    cv2.putText(result_image, f'there is a wall thickness error in tupe number {error_list}', (10,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2)
if  not np.all(ce):
    cv2.putText(clineImage, 'there is an edge thickness error', (10,90), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2)
    cv2.putText(result_image, 'there is an edge thickness error', (10,90), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2)
else:
    cv2.putText(clineImage, 'No wall thickness error detected', (0,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2)
    cv2.putText(result_image, 'No wall thickness error detected', (0,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2)

cv2.imshow('res filtered', cv2.resize(clineImage, (800, 600)))
cv2.imshow('Final Result', cv2.resize(result_image, (800, 600)))
cv2.imwrite(os.path.join(results_path, 'wall_thickness 2.jpg'), cv2.resize(clineImage, (800, 600)))
cv2.imwrite(os.path.join(results_path, 'final result.jpg'), cv2.resize(result_image, (800, 600)))

# print(check_thickness(filtered_lines, 5).index(True))
# cv2.imwrite(os.path.join(results_path, 'res filtered.jpg'), cv2.resize(clineImage, (800, 600)))


cv2.waitKey(0)
cv2.destroyAllWindows()