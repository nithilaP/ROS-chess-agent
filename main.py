import pyrealsense2 as rs
import numpy as np
import cv2
from sklearn.cluster import KMeans
from edge_detection_utils import *
from chess_engine import *
import chess

# Define global variables

# RED HSV Value Calibration
MIN_RED_HSV_MASK1 = (0, 70, 50)
MAX_RED_HSV_MASK1 = (10, 255, 255)
MIN_RED_HSV_MASK2 = (170, 70, 50)
MAX_RED_HSV_MASK2 = (180, 255, 255)

LINE_ANGLE_CUTOFF = 9
WIDTH_SQUARE = 8
HEIGHT_SQUARE = 8
MULTIPLIER_LENGTH = 100
INTERSECT_PTS_LAST = None

OFFSET = int(0.5 * MULTIPLIER_LENGTH) # 0.6
SQUARE_SIZE = int(0.5 * MULTIPLIER_LENGTH) # 0.3
CIRCLE_PIXELS = int(0.93 * MULTIPLIER_LENGTH)

BOARD_DISTANCE = 0

def video_capture():

    global MIN_RED_HSV_MASK1 
    global MAX_RED_HSV_MASK1 
    global MIN_RED_HSV_MASK2 
    global MAX_RED_HSV_MASK2 

    global LINE_ANGLE_CUTOFF
    global WIDTH_SQUARE
    global HEIGHT_SQUARE
    global MULTIPLIER_LENGTH
    global INTERSECT_PTS_LAST

    global OFFSET
    global SQUARE_SIZE
    global CIRCLE_PIXELS

    global BOARD_DISTANCE

    # Pipeline to get the data stream from the camera -> configuring camera
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    
    last_state = [["black"] * 8,
                  ["black"] * 8,
                  [None] * 8,
                  [None] * 8,
                  [None] * 8,
                  [None] * 8,
                  ["white"] * 8,
                  ["white"] * 8]
    last_state = np.array(last_state)

    last_state_mask = (last_state != None)
    print(last_state_mask)
    board = chess.Board()

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

            # Get HSV values for tuning
            # mean_center_hsv = np.mean(hsv[len(hsv)//2], axis=0)
            # print('Current Mean Value:', mean_center_hsv)
            # mask the red color of the board -> https://stackoverflow.com/questions/32522989/opencv-better-detection-of-red-color
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

                h, status = cv2.findHomography(intersect_pts, np.array([[0, 0], [WIDTH_SQUARE*MULTIPLIER_LENGTH, 0], [WIDTH_SQUARE*MULTIPLIER_LENGTH, HEIGHT_SQUARE*MULTIPLIER_LENGTH], [0, HEIGHT_SQUARE*MULTIPLIER_LENGTH]]))
                im_dst = cv2.warpPerspective(color_image, h, (WIDTH_SQUARE*MULTIPLIER_LENGTH,  HEIGHT_SQUARE*MULTIPLIER_LENGTH))
                # im_depth = cv2.warpPerspective(depth_image, h, (WIDTH_SQUARE*MULTIPLIER_LENGTH,  HEIGHT_SQUARE*MULTIPLIER_LENGTH))


                cv2.imwrite("cleared.png", im_dst)


                # Image Frame Processing -> Get & Analyze Gamestate
                current_state = np.array([[None] * WIDTH_SQUARE] * HEIGHT_SQUARE)            
                for i in range(WIDTH_SQUARE):
                    for j in range(HEIGHT_SQUARE):
                        # cv2.circle(output[0], (65 + i * 93, 65 + 93 * j), radius=5, thickness=-1, color=(0, 255, 0))
                        # cv2.circle(output[0], (70 + i * 93, 70 + 93 * j), radius=5, thickness=-1, color=(0, 255, 0))
                        min_x = OFFSET + i * CIRCLE_PIXELS
                        max_x = min_x + SQUARE_SIZE
                        min_y = OFFSET + j * CIRCLE_PIXELS
                        max_y = min_y + SQUARE_SIZE
                        cv2.rectangle(im_dst, (min_x, min_y), (max_x, max_y), color=(0, 255, 0), thickness=2)
                        patch = im_dst[min_y:max_y, min_x:max_x]
                        # print(patch.shape)

                        chess_loc = convert_to_chess_loc(i, j)
                        current_detection = piece_detection(patch, im_dst)
                        current_state[i, j] = current_detection
                        if current_detection != None:
                            print(f"{current_detection} piece detected at", chess_loc)
                        
                        # Using cv2.putText() method
                        im_dst = cv2.putText(im_dst, chess_loc, (int(max_x - SQUARE_SIZE/ 2), int(max_y - SQUARE_SIZE/ 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 0, 0), thickness=1)
                        
                cv2.imwrite("outlined_gamestate.png", im_dst)
                current_state_mask = (current_state != None)
                player_move = get_player_move(last_state_mask, current_state_mask, board, current_state)
                print("PLAYER MOVE:", player_move)
                if board.is_checkmate() or board.is_stalemate():
                    print("GAME OVER - PLAYER WON")
                    break
                board.push_san(player_move)
                robot_move = get_best_move(board, "black")
                print("ROBOT MOVE:", robot_move)
                # send commands to Arduino function -> wait for response
                board.push_san(robot_move)
                last_state_mask = current_state_mask
                last_state = current_state

                if board.is_checkmate() or board.is_stalemate():
                    print("GAME OVER - ROBOT WON")
                    break
                
    finally:
        pipeline.stop()
        print("DONE")

if __name__ == "__main__":
    video_capture()