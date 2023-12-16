import pyrealsense2 as rs
import numpy as np
import cv2

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

def cyclic_intersection_pts(pts):
    if pts.shape[0] != 4:
        return None

    # Calculate the center
    center = np.mean(pts, axis=0)

    # Sort the points in clockwise
    cyclic_pts = [
        # Top-left
        pts[np.where(np.logical_and(pts[:, 0] < center[0], pts[:, 1] < center[1]))[0][0], :],
        # Top-right
        pts[np.where(np.logical_and(pts[:, 0] > center[0], pts[:, 1] < center[1]))[0][0], :],
        # Bottom-Right
        pts[np.where(np.logical_and(pts[:, 0] > center[0], pts[:, 1] > center[1]))[0][0], :],
        # Bottom-Left
        pts[np.where(np.logical_and(pts[:, 0] < center[0], pts[:, 1] > center[1]))[0][0], :]
    ]

    return np.array(cyclic_pts)

def drawHoughLines(image, lines, output):
    out = image.copy()
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
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(output, out)

def intersection(m1: float, b1: float, m2: float, b2: float):
    # Consider y to be equal and solve for x
    # Solve:
    #   m1 * x + b1 = m2 * x + b2
    if b2 is np.nan or b1 is np.nan or m1 is np.nan or m2 is np.nan or m1 - m2 == 0:# 0.0000000001:
        return None, None
    x = (b2 - b1) / (m1 - m2)
    # Use the value of x to calculate y
    y = m1 * x + b1

    return int(round(x)), int(round(y))

def hough_lines_intersection(lines: np.array, image_shape: tuple):
    """
    Returns the intersection points that lie on the image
    for all combinations of the lines
    """
    if len(lines.shape) == 3 and \
            lines.shape[1] == 1 and lines.shape[2] == 2:
        lines = np.squeeze(lines)
    lines_count = len(lines)
    intersect_pts = []
    for i in range(lines_count - 1):
        for j in range(i + 1, lines_count):
            # print(lines[i])
            m1, b1 = polar2cartesian(lines[i][0], lines[i][1], True)
            m2, b2 = polar2cartesian(lines[j][0], lines[j][1], True)
            x, y = intersection(m1, b1, m2, b2)
            # print(x, y)
            if x is not None and y is not None and point_on_image(x, y, image_shape):
                intersect_pts.append([x, y])
                # print("appended")
    return np.array(intersect_pts, dtype=int)


def polar2cartesian(rho: float, theta_rad: float, rotate90: bool = False):
    """
    Converts line equation from polar to cartesian coordinates
    Args:
        rho: input line rho
        theta_rad: input line theta
        rotate90: output line perpendicular to the input line
    Returns:
        m: slope of the line
           For horizontal line: m = 0
           For vertical line: m = np.nan
        b: intercept when x=0
    """
    x = np.cos(theta_rad) * rho
    y = np.sin(theta_rad) * rho
    m = np.nan
    if not np.isclose(x, 0.0):
        m = y / x
    if rotate90:
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

def point_on_image(x: int, y: int, image_shape: tuple):
    """
    Returns true is x and y are on the image
    """
    return 0 <= y < image_shape[0] and 0 <= x < image_shape[1]

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

    # convert to white and black frame
    green = cv2.bitwise_and(blanks_green, blanks_green, mask=mask1)
    # black_out2 = cv2.bitwise_and(blanks, blanks, mask=mask2)
    # black_out = cv2.bitwise_or(black_out1, black_out2)
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

