import pyrealsense2 as rs
import numpy as np
import cv2
from sklearn.cluster import KMeans

LETTERS =  ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

MIN_GREEN_HSV_MASK = (50, 100, 30)
MAX_GREEN_HSV_MASK = (70, 255, 255)

MIN_BLUE_HSV_MASK = (90, 50, 70)
MAX_BLUE_HSV_MASK = (128, 255,255)

def convert_to_chess_loc(i, j):
    global LETTERS
    return LETTERS[i] + str(8 - j)

def convert_to_2d(chess_loc):
    return (LETTERS.index(chess_loc[0]), 8 - int(chess_loc[1]))

def display_hough_lines(image, lines, output):
    hough_lines = image.copy()
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))
        cv2.line(hough_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(output, hough_lines)

def get_intersect_pt(m1, b1, m2, b2):
    if b2 is np.nan or b1 is np.nan or m1 is np.nan or m2 is np.nan or m1 - m2 == 0:
        return None, None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    return int(round(x)), int(round(y))

def get_intersect_pts(lines, image):
    if len(lines.shape) == 3 and lines.shape[1] == 1 and lines.shape[2] == 2:
        lines = np.squeeze(lines)
    num_lines = len(lines)
    intersect_pts = []
    for i in range(num_lines - 1):
        for j in range(i + 1, num_lines):
            i_rho, i_theta = lines[i]
            j_rho, j_theta = lines[j]
            m1, b1 = get_slope_intersect(i_rho, i_theta)
            m2, b2 = get_slope_intersect(j_rho, j_theta)
            x, y = get_intersect_pt(m1, b1, m2, b2)
            if x is not None and y is not None and is_on_image(x, y, image.shape):
                intersect_pts.append([x, y])
    intersect_pts = np.array(intersect_pts, dtype=int)

    if intersect_pts.shape[0] != 4:
        return None

    x_center, y_center = np.mean(intersect_pts, axis=0)

    organized_intersect_pts = [0] * 4

    organized_intersect_pts[0] = intersect_pts[np.where(np.logical_and(intersect_pts[:, 0] < x_center,
                                                                        intersect_pts[:, 1] < y_center))[0][0], :]
    organized_intersect_pts[1] = intersect_pts[np.where(np.logical_and(intersect_pts[:, 0] > x_center,
                                                                        intersect_pts[:, 1] < y_center))[0][0], :]
    organized_intersect_pts[2] = intersect_pts[np.where(np.logical_and(intersect_pts[:, 0] > x_center,
                                                                        intersect_pts[:, 1] > y_center))[0][0], :]
    organized_intersect_pts[3] = intersect_pts[np.where(np.logical_and(intersect_pts[:, 0] < x_center,
                                                                        intersect_pts[:, 1] > y_center))[0][0], :]

    return np.array(organized_intersect_pts)


def get_slope_intersect(rho, theta_rad):
    x = np.cos(theta_rad) * rho
    y = np.sin(theta_rad) * rho
    m = np.nan
    if not np.isclose(x, 0.0):
        m = y / x
        if m is np.nan:
            m = 0.0
        elif np.isclose(m, 0.0):
            m = np.nan
        else:
            m = -1.0 / m
    b = 0.0
    if m is not np.nan:
        b = y - m * x
    
    return m, b

def is_on_image(x, y, image_shape):
    img_y, img_x = image_shape
    return 0 <= y < img_y and 0 <= x < img_x

def piece_detection(patch_image, im_dst):
    global MIN_BLUE_HSV_MASK
    global MAX_BLUE_HSV_MASK

    global MIN_GREEN_HSV_MASK
    global MAX_GREEN_HSV_MASK

    hsv = cv2.cvtColor(im_dst, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, MIN_GREEN_HSV_MASK, MAX_GREEN_HSV_MASK)
    mask2 = cv2.inRange(hsv, MIN_BLUE_HSV_MASK, MAX_BLUE_HSV_MASK)
    blanks_green = np.ones_like(im_dst) * 255
    blanks_blue = np.ones_like(im_dst) * 255

    green = cv2.bitwise_and(blanks_green, blanks_green, mask=mask1)
    green_outline = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
    blue = cv2.bitwise_and(blanks_blue, blanks_blue, mask=mask2)
    blue_outline = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("green.png", green_outline)
    cv2.imwrite("blue.png", blue_outline)


    patch_color = np.mean(patch_image, (0, 1)).reshape((1, 1, 3)).astype("uint8")
    patch_color = cv2.cvtColor(patch_color, cv2.COLOR_BGR2HSV)
    is_white = cv2.inRange(patch_color, MIN_BLUE_HSV_MASK, MAX_BLUE_HSV_MASK)[0]
    is_black = cv2.inRange(patch_color, MIN_GREEN_HSV_MASK, MAX_GREEN_HSV_MASK)[0]
    if is_black:
        return "black"
    if is_white:
        return "white"
    return None


def get_hough_lines(contour_outline, angle_cutoff):
    hough_lines = cv2.HoughLines(contour_outline, 1, np.pi / 180, 300, min_theta=1e-9)
    hough_lines_squeeze = np.squeeze(hough_lines)
    cutoff = np.pi/angle_cutoff
    hough_lines = hough_lines[(hough_lines_squeeze[:, 1] < cutoff) | 
                              (hough_lines_squeeze[:, 1] > (angle_cutoff-1)*cutoff) | 
                              ((hough_lines_squeeze[:, 1] > (angle_cutoff-1)*cutoff/2) & 
                               (hough_lines_squeeze[:, 1] < (angle_cutoff+1)*cutoff/2))]
    hough_lines[hough_lines[:, 0, 1] > (angle_cutoff-1)*np.pi/cutoff, 0, 0] *= -1
    hough_lines[hough_lines[:, 0, 1] > (angle_cutoff-1)*np.pi/cutoff, 0, 1] -= np.pi
    kmeans = KMeans(n_clusters=4, n_init=10, max_iter=300).fit(hough_lines[:, 0, :])
    hough_lines =  np.expand_dims(kmeans.cluster_centers_, axis=1)
    return hough_lines
