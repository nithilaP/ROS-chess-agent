import pyrealsense2 as rs
import numpy as np
import cv2
from sklearn.cluster import KMeans

# Define global variables

# RED HSV Value Calibration
MIN_RED_HSV_MASK1 = (0, 70, 50)
MAX_RED_HSV_MASK1 = (10, 255, 255)
MIN_RED_HSV_MASK2 = (170, 70, 50)
MAX_RED_HSV_MASK2 = (180, 255, 255)

LINE_ANGLE_CUTOFF = 9
WIDTH_CIRCLES = 7
HEIGHT_CIRCLES = 6
MULTIPLIER_LENGTH = 100
INTERSECT_PTS_LAST = None

def cyclic_intersection_pts(pts):
    """
    Sorts 4 points in clockwise direction with the first point been closest to 0,0
    Assumption:
        There are exactly 4 points in the input and
        from a rectangle which is not very distorted
    """
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
            print(lines[i])
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

def video_capture():

    global MIN_RED_HSV_MASK1 
    global MAX_RED_HSV_MASK1 
    global MIN_RED_HSV_MASK2 
    global MAX_RED_HSV_MASK2 
    global LINE_ANGLE_CUTOFF
    global WIDTH_CIRCLES
    global HEIGHT_CIRCLES
    global MULTIPLIER_LENGTH
    global INTERSECT_PTS_LAST

    # Pipeline to get the data stream from the camera -> configuring camera
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Get HSV values for tuning
            # mean_center_hsv = np.mean(hsv[len(hsv)//2], axis=0)
            # print('Current Mean Value:', mean_center_hsv)
            # mask the red color of the board
            mask1 = cv2.inRange(hsv, MIN_RED_HSV_MASK1, MAX_RED_HSV_MASK1)
            mask2 = cv2.inRange(hsv, MIN_RED_HSV_MASK2, MAX_RED_HSV_MASK2)
            blanks = np.ones_like(color_image) * 255

            # convert to white and black frame
            black_out1 = cv2.bitwise_and(blanks, blanks, mask=mask1)
            black_out2 = cv2.bitwise_and(blanks, blanks, mask=mask2)
            black_out = cv2.bitwise_or(black_out1, black_out2)
            black_white = cv2.cvtColor(black_out, cv2.COLOR_BGR2GRAY)

            # contours
            contours, hierarchy = cv2.findContours(black_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            try: # does following code when it sees red in the frame
                contour_idx = np.argmax([cv2.contourArea(cont) for cont in contours])
                cv2.drawContours(color_image, contours, contour_idx, (255, 255, 255), 3)
                contour_image = np.zeros_like(color_image)
                cv2.drawContours(contour_image, contours, contour_idx, (255, 255, 255), 3)
                contour_outline_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
            except ValueError:
                pass

            # cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Color", color_image)
            cv2.imshow("Depth", depth_colormap)
            # cv2.imshow("Depth", depth_image)
            keystroke = cv2.waitKey(1)
            if keystroke == ord('q'):
                break
            if keystroke == ord(" "):
                cv2.imwrite("contours_color.png", color_image)
                cv2.imwrite("contours_depth.png", depth_image)
                print("TOOK A SNAPSHOT")
                cv2.imwrite("contour_outline_img.png", contour_outline_image)
            
                # Hough Lines
                try:
                        # blank = blank[CROP:-CROP, CROP:-CROP] # Can crop the contour outline image
                    polar_lines = cv2.HoughLines(contour_outline_image, 1, np.pi / 180, 300, min_theta=1e-9)
                    polar_lines_squeeze = np.squeeze(polar_lines)

                    LINE_ANGLE_CUTOFF = 9
                    polar_lines = polar_lines[(polar_lines_squeeze[:, 1] < np.pi/LINE_ANGLE_CUTOFF) | (polar_lines_squeeze[:, 1] > (LINE_ANGLE_CUTOFF-1)*np.pi/LINE_ANGLE_CUTOFF) | ((polar_lines_squeeze[:, 1] > (LINE_ANGLE_CUTOFF-1)*np.pi/LINE_ANGLE_CUTOFF/2) &  (polar_lines_squeeze[:, 1] < (LINE_ANGLE_CUTOFF+1)*np.pi/LINE_ANGLE_CUTOFF/2))]
                    polar_lines[polar_lines[:, 0, 1] > (LINE_ANGLE_CUTOFF-1)*np.pi/LINE_ANGLE_CUTOFF, 0, 0] *= -1
                    polar_lines[polar_lines[:, 0, 1] > (LINE_ANGLE_CUTOFF-1)*np.pi/LINE_ANGLE_CUTOFF, 0, 1] -= np.pi
                
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=4, n_init=10, max_iter=300).fit(polar_lines[:, 0, :])
                    polar_lines =  np.expand_dims(kmeans.cluster_centers_, axis=1)
                    drawHoughLines(color_image, polar_lines, 'hough_lines.png')

                    # Detect the intersection points
                    intersect_pts = hough_lines_intersection(polar_lines, contour_image.shape)
                    # Sort the points in cyclic order
                    intersect_pts = cyclic_intersection_pts(intersect_pts)
                    if intersect_pts is None:
                        raise Exception("Intersections not found") 

                    INTERSECT_PTS_LAST = intersect_pts
                    for pt in intersect_pts:
                        cv2.circle(color_image, (pt[0], pt[1]), radius=5, thickness=-1, color=(0, 255, 0))
                    cv2.imwrite('intersections.png',color_image) 
                    
                except Exception as e: 
                    print("Homography failed...")
                    print(e)
                    intersect_pts = INTERSECT_PTS_LAST
                    if intersect_pts is None:
                        continue

                h, status = cv2.findHomography(intersect_pts, np.array([[0, 0], [WIDTH_CIRCLES*MULTIPLIER_LENGTH, 0], [WIDTH_CIRCLES*MULTIPLIER_LENGTH, HEIGHT_CIRCLES*MULTIPLIER_LENGTH], [0, HEIGHT_CIRCLES*MULTIPLIER_LENGTH]]))
                im_dst = cv2.warpPerspective(color_image, h, (WIDTH_CIRCLES*MULTIPLIER_LENGTH,  HEIGHT_CIRCLES*MULTIPLIER_LENGTH))

                cv2.imwrite("hough_lines.png", im_dst)
                
    finally:
        pipeline.stop()

if __name__ == "__main__":
    video_capture()