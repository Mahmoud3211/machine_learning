import cv2
import numpy as np
import os
from PIL import Image
class wall_thickness_detector:
    
    def __init__(self, empty_image):
        self.empty_image = np.array(empty_image)
        
    
    def slope_intercept(self, x1,y1,x2,y2):
        """
        This function returns the slope and intercept of any two givin points.
        """
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        return a,b

    def sort_lines(self, lines):
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

    def get_points_on_line(self, points):
        slope, intercept = self.get_slop_intercept(points)
        if slope == None:
            return None, None
        xs = np.array(range(1024))
        ys = xs * slope + intercept
        return xs.astype(int), ys.astype(int)

    def get_slop_intercept(self, points):
        x1, y1, x2, y2 = points
        slope = (y2 - y1) // (x2 - x1) if (x2 - x1) != 0 else None
        if slope == None:
            return None, None
        intercept = y1 - (slope * x1)
        return int(slope) , int(intercept)

    def filter_irrelevant_lines(self, lines):
        new_lines = []
        for line in lines:
            if (np.abs(1024 - line[0]) > 1015 or np.abs(1024 - line[0]) < 10) and (np.abs(1024 - line[2]) > 1015 or np.abs(1024 - line[2]) < 10):
                continue
            if (np.abs(1024 - line[1]) > 1015 or np.abs(1024 - line[1]) < 10) and (np.abs(1024 - line[3]) > 1015 or np.abs(1024 - line[3]) < 10):
                continue
            else:
                new_lines.append(line)
        return new_lines

    def filter_with_point(self, lines, y=200, tx=10):
        selected_y = y
        lines = self.filter_irrelevant_lines(self.sort_lines(lines))
        x = 0
        info_array = []
        for line in lines:

            slope, intercept = self.get_slop_intercept(line)
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
            
    def check_thickness(self, full_lines, diff_thresh=3, edge_thresh=3):
        ys = [300, 500, 700, 900]

        selected_coor = []
        check_array = []
        edges_array = []
        # Handle the case where the split generates an element with lenght other than 4!
        for group in np.array_split(full_lines, len(full_lines) // 4):
            
            for line in group:
                slope, intercept = self.get_slop_intercept(line)
                slope = 0 if slope is None else slope
                intercept = 0 if intercept is None else intercept
                
                x1 = (ys[0] - intercept) // slope if slope != 0 else 0
                x2 = (ys[1] - intercept) // slope if slope != 0 else 0
                x3 = (ys[2] - intercept) // slope if slope != 0 else 0
                x4 = (ys[3] - intercept) // slope if slope != 0 else 0

                selected_coor.append(np.array([x1, x2, x3, x4]))
            dist1 = np.abs(selected_coor[0] - selected_coor[1])
            dist2 = np.abs(selected_coor[2] - selected_coor[3])

            edge_diff = np.all(np.abs(dist1 - dist2) >= edge_thresh)
            right_edge = np.all(dist1.max() - dist1.min() >= diff_thresh)
            left_edge = np.all(dist2.max() - dist2.min() >= diff_thresh)

            edges_array.append(right_edge)
            edges_array.append(left_edge)
            check_array.append(edge_diff)
            selected_coor = []

        return check_array, edges_array

    def get_single_point_on_line(self, line, selected_y):
        slope, intercept = self.get_slop_intercept(line)
        slope = 0 if slope is None else slope
        intercept = 0 if intercept is None else intercept
        x = (selected_y - intercept) // slope if slope != 0 else 0
        return [x, selected_y]

    def draw_flaws(self, img, filtered_lines, checked_pipes, checked_edges):
        flawed_pips = [i for i, c in enumerate(checked_pipes) if c]
        flawed_edges = [i for i, c in enumerate(checked_edges) if c]

        flawed_groups = np.array(np.array_split(filtered_lines, len(filtered_lines) // 4))[flawed_pips]
        flawed_edges =  np.array(np.array_split(filtered_lines, len(filtered_lines) // 2))[flawed_edges]
        
        overlay = img.copy()
        overlay2 = img.copy()
        for group in flawed_groups:
            p11 = self.get_single_point_on_line(group[0], 150)
            p12 = self.get_single_point_on_line(group[0], 1000)
            p13 = self.get_single_point_on_line(group[1], 1000)
            p14 = self.get_single_point_on_line(group[1], 150)

            p21 = self.get_single_point_on_line(group[2], 150)
            p22 = self.get_single_point_on_line(group[2], 1000)
            p23 = self.get_single_point_on_line(group[3], 1000)
            p24 = self.get_single_point_on_line(group[3], 150)

            poly_points_1 = np.array([p11, p12, p13, p14])
            poly_points_2 = np.array([p21, p22, p23, p24])
            cv2.fillPoly(overlay,[poly_points_1], (255,0,0))        
            cv2.fillPoly(overlay,[poly_points_2], (255,0,0)) 

        for edge in flawed_edges:
            p11 = self.get_single_point_on_line(edge[0], 150)
            p12 = self.get_single_point_on_line(edge[0], 1000)
            p13 = self.get_single_point_on_line(edge[1], 1000)
            p14 = self.get_single_point_on_line(edge[1], 150)
            
            poly_points_1 = np.array([p11, p12, p13, p14])
            cv2.fillPoly(overlay2,[poly_points_1], (0,0,255))
        
        alpha = 0.7
        alpha1 = 0.5
        new_empty = cv2.addWeighted(overlay, alpha1, overlay2, 1 - alpha1, 0, overlay2)
        cv2.addWeighted(new_empty, alpha, img, 1 - alpha, 0, img)
        
        # cv2.imwrite('./empty image poly.jpg', cv2.resize(img, (800, 600)))
        message = ""
        error_list = [i + 1 for i, c in enumerate(checked_pipes) if c]
        messages_list = []
        if len(error_list):
            message = f'there is a wall thickness error in tupe number {error_list}'
            messages_list.append(message)                    
        if True in checked_edges:
            edges = [i + 1 for i, e in enumerate(checked_edges) if e]
            message = f'there is an edge thickness error on edges {edges}'
            messages_list.append(message)                    
            
        else:
            message = 'No wall thickness error detected'
            messages_list.append(message)                    
            
        return Image.fromarray(img), messages_list

    def detect(self):

        de_image = cv2.bilateralFilter(self.empty_image, 9, 75, 75)
        gray = cv2.cvtColor(de_image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 255)
        kernel = np.ones((5,5), np.uint8)
        img_dilation = cv2.dilate(edged, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(img_dilation,
            cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        lines = cv2.HoughLinesP(img_dilation, 1, np.pi/500, 50, np.array([]), 300, 0)

        new_lines = self.sort_lines(lines)

        filtered_lines = self.filter_with_point(lines, 512, 20)

        ct, ce = self.check_thickness(filtered_lines, 4, 3)

        return self.draw_flaws(self.empty_image, filtered_lines, ct, ce)


