import cv2
import numpy as np
import os

def slope_intercept(x1,y1,x2,y2):
    """
    This function returns the slope and intercept of any two givin points.
    """
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
        if (np.abs(1024 - line[0]) > 1015 or np.abs(1024 - line[0]) < 10) and (np.abs(1024 - line[2]) > 1015 or np.abs(1024 - line[2]) < 10):
            continue
        if (np.abs(1024 - line[1]) > 1015 or np.abs(1024 - line[1]) < 10) and (np.abs(1024 - line[3]) > 1015 or np.abs(1024 - line[3]) < 10):
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

        slope, intercept = get_slop_intercept(line)
        slope = 0 if slope is None else slope
        intercept = 0 if intercept is None else intercept
        x = (selected_y - intercept) // slope if slope != 0 else 0
        info_array.append(x)

    lines = np.array(lines)[np.argsort(info_array)]
    info_array = np.array(info_array)[np.argsort(info_array)]
    new_array = []
    index_array = []
    for i, x in enumerate(info_array):
        if not new_array or abs(x - new_array[-1]) > tx:
            new_array.append(x)
            index_array.append(i)

    return np.array(lines)[index_array]
        
def check_thickness(full_lines, diff_thresh=3, edge_thresh=3):
    ys = [300, 500, 700, 900]

    selected_coor = []
    check_array = []
    edges_array = []
    # Handle the case where the split generates an element with lenght other than 4!
    for group in np.array_split(full_lines, len(full_lines) // 4):
        
        for line in group:
            slope, intercept = get_slop_intercept(line)
            slope = 0 if slope is None else slope
            intercept = 0 if intercept is None else intercept
            
            x1 = (ys[0] - intercept) // slope if slope != 0 else 0
            x2 = (ys[1] - intercept) // slope if slope != 0 else 0
            x3 = (ys[2] - intercept) // slope if slope != 0 else 0
            x4 = (ys[3] - intercept) // slope if slope != 0 else 0

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
        selected_coor = []

    return check_array, edges_array

def get_single_point_on_line(line, selected_y):
    slope, intercept = get_slop_intercept(line)
    slope = 0 if slope is None else slope
    intercept = 0 if intercept is None else intercept
    x = (selected_y - intercept) // slope if slope != 0 else 0
    return [x, selected_y]

def draw_flaws(img, filtered_lines, checked_pipes, checked_edges):
    flawed_pips = [i for i, c in enumerate(checked_pipes) if c]
    flawed_edges = [i for i, c in enumerate(checked_edges) if c]

    flawed_groups = np.array(np.array_split(filtered_lines, len(filtered_lines) // 4))[flawed_pips]
    flawed_edges =  np.array(np.array_split(filtered_lines, len(filtered_lines) // 2))[flawed_edges]
    
    overlay = img.copy()
    overlay2 = img.copy()
    for group in flawed_groups:
        p11 = get_single_point_on_line(group[0], 150)
        p12 = get_single_point_on_line(group[0], 1000)
        p13 = get_single_point_on_line(group[1], 1000)
        p14 = get_single_point_on_line(group[1], 150)

        p21 = get_single_point_on_line(group[2], 150)
        p22 = get_single_point_on_line(group[2], 1000)
        p23 = get_single_point_on_line(group[3], 1000)
        p24 = get_single_point_on_line(group[3], 150)

        poly_points_1 = np.array([p11, p12, p13, p14])
        poly_points_2 = np.array([p21, p22, p23, p24])
        cv2.fillPoly(overlay,[poly_points_1], (0,0,255))        
        cv2.fillPoly(overlay,[poly_points_2], (0,0,255))      

    for edge in flawed_edges:
        p11 = get_single_point_on_line(edge[0], 150)
        p12 = get_single_point_on_line(edge[0], 1000)
        p13 = get_single_point_on_line(edge[1], 1000)
        p14 = get_single_point_on_line(edge[1], 150)
        
        poly_points_1 = np.array([p11, p12, p13, p14])
        cv2.fillPoly(overlay2,[poly_points_1], (255,0,0))
    
    alpha = 0.7
    alpha1 = 0.5
    new = cv2.addWeighted(overlay, alpha1, overlay2, 1 - alpha1, 0, overlay2)
    cv2.addWeighted(new, alpha, img, 1 - alpha, 0, img)
    cv2.imshow('poly', cv2.resize(img, (800, 600)))
    cv2.imwrite(os.path.join(results_path, 'final_all_classes.jpg'), cv2.resize(img, (800, 600)))
    


results_path = os.path.join(os.getcwd(), 'results')
test_path = os.path.join(os.getcwd(), 'test')

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

new_lines = sort_lines(lines)
for x1, y1, x2, y2  in new_lines:
    xs, ys = get_points_on_line((x1, y1, x2, y2))
    if xs is not None:
        cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for x, y in zip(xs, ys):
            cv2.circle(lineImage, (x,y), 2, (0,0,255), 2)
# cv2.imshow('res', cv2.resize(lineImage, (800, 600)))
# cv2.imwrite(os.path.join(results_path, 'res.jpg'), cv2.resize(lineImage, (800, 600)))
# cv2.waitKey(0)
# quit()
# filtered_lines = m_filter_with_slopes(lines, 0.99, 1.05, 0.99, 1.05)
filtered_lines = filter_with_point(lines, 512, 20)
print('[--]', len(filtered_lines))

for x1, y1, x2, y2  in filtered_lines:
    xs, ys = get_points_on_line((x1, y1, x2, y2))
    cv2.line(clineImage, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if xs is not None:
        for x, y in zip(xs, ys):
            cv2.circle(clineImage, (x,y), 2, (255,0,0), 2)
# cv2.imshow('res filtered', cv2.resize(clineImage, (800, 600)))
# cv2.waitKey(0)
# quit()
error_list = []
ct, ce = check_thickness(filtered_lines, 4, 3)
print(ct, ce)

draw_flaws(result_image, filtered_lines, ct, ce)

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