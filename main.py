import pyrealsense2 as rs
import numpy as np
import cv2
from edge_detection_utils import *
from chess_engine import *
import chess
import serial
from solver_to_gantry import *

MIN_RED_HSV_MASK1 = (0, 70, 50)
MAX_RED_HSV_MASK1 = (10, 255, 255)
MIN_RED_HSV_MASK2 = (170, 70, 50)
MAX_RED_HSV_MASK2 = (180, 255, 255)

WIDTH_SQUARE = 8
HEIGHT_SQUARE = 8
MULTIPLIER = 100
PREV_INTERSECT_PTS = None

OFFSET = int(0.5 * MULTIPLIER)
SQUARE_SIZE = int(0.5 * MULTIPLIER)
BOX_PIXELS = int(0.93 * MULTIPLIER)


# Definition for Arduino
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1) 

def video_capture():

    global MIN_RED_HSV_MASK1 
    global MAX_RED_HSV_MASK1 
    global MIN_RED_HSV_MASK2 
    global MAX_RED_HSV_MASK2 

    global ANGLE_CUTOFF
    global WIDTH_SQUARE
    global HEIGHT_SQUARE
    global MULTIPLIER
    global PREV_INTERSECT_PTS

    global OFFSET
    global SQUARE_SIZE
    global BOX_PIXELS

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
    board = chess.Board()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            mask1 = cv2.inRange(hsv, MIN_RED_HSV_MASK1, MAX_RED_HSV_MASK1)
            mask2 = cv2.inRange(hsv, MIN_RED_HSV_MASK2, MAX_RED_HSV_MASK2)
            blanks = np.ones_like(color_image) * 255

            black_out1 = cv2.bitwise_and(blanks, blanks, mask=mask1)
            black_out2 = cv2.bitwise_and(blanks, blanks, mask=mask2)
            black_out = cv2.bitwise_or(black_out1, black_out2)
            black_white = cv2.cvtColor(black_out, cv2.COLOR_BGR2GRAY)

            contours, hierarchy = cv2.findContours(black_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            try: 
                contour_idx = np.argmax([cv2.contourArea(cont) for cont in contours])
                cv2.drawContours(color_image, contours, contour_idx, (255, 255, 255), 3)
                contour_image = np.zeros_like(color_image)
                cv2.drawContours(contour_image, contours, contour_idx, (255, 255, 255), 3)
                contour_outline_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
            except ValueError:
                pass

            cv2.imshow("Color", color_image)
            keystroke = cv2.waitKey(1)
            if keystroke == ord('q'):
                break
            if keystroke == ord(" "):
                cv2.imwrite("contours_color.png", color_image)
                cv2.imwrite("contour_outline_img.png", contour_outline_image)
            
                try:
                    hough_lines = get_hough_lines(contour_outline_image, angle_cutoff=9)
                    display_hough_lines(color_image, hough_lines, 'hough_lines.png')

                    intersect_pts = get_intersect_pts(hough_lines, contour_image)
                    if intersect_pts is None:
                        raise Exception("Intersection points not found") 

                    PREV_INTERSECT_PTS = intersect_pts
                    for pt in intersect_pts:
                        x, y = pt
                        cv2.circle(color_image, (x, y), radius=5, thickness=-1, color=(0, 255, 0))
                    cv2.imwrite('intersections.png',color_image) 
                    
                except Exception as e: 
                    print("Failed to Properly Parse Board")
                    print(e)
                    intersect_pts = PREV_INTERSECT_PTS
                    if intersect_pts is None:
                        continue

                h, status = cv2.findHomography(intersect_pts,
                                                np.array([[0, 0], [WIDTH_SQUARE*MULTIPLIER, 0], [WIDTH_SQUARE*MULTIPLIER, HEIGHT_SQUARE*MULTIPLIER], [0, HEIGHT_SQUARE*MULTIPLIER]]))
                im_dst = cv2.warpPerspective(color_image, h, (WIDTH_SQUARE*MULTIPLIER,  HEIGHT_SQUARE*MULTIPLIER))


                cv2.imwrite("cleared.png", im_dst)


                current_state = np.array([[None] * WIDTH_SQUARE] * HEIGHT_SQUARE)            
                for i in range(WIDTH_SQUARE):
                    for j in range(HEIGHT_SQUARE):
                        min_x = OFFSET + i * BOX_PIXELS
                        max_x = min_x + SQUARE_SIZE
                        min_y = OFFSET + j * BOX_PIXELS
                        max_y = min_y + SQUARE_SIZE
                        cv2.rectangle(im_dst, (min_x, min_y), (max_x, max_y), color=(0, 255, 0), thickness=2)
                        patch = im_dst[min_y:max_y, min_x:max_x]

                        chess_loc = convert_to_chess_loc(i, j)
                        current_detection = piece_detection(patch, im_dst)
                        current_state[i, j] = current_detection
                        if current_detection != None:
                            print(f"{current_detection} piece detected at", chess_loc)
                        
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
                # check if something is there 
                from_loc, to_loc = robot_move[:2], robot_move[2:]
                i, j = convert_to_2d(to_loc)
                if (current_state_mask[i][j] == 0):
                    write_move(to_loc)
                    write_move("taken")
                write_move(from_loc)
                write_move(to_loc)
                write_move("origin")

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