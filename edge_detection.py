import pyrealsense2 as rs
import numpy as np
import cv2

def video_capture():
    MIN_RED_HSV = (0, 70, 90)
    MAX_RED_HSV = (40, 255, 210)

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

            # Get HSV values for tuning
            # mean_center_hsv = np.mean(hsv[len(hsv)//2], axis=0)
            # print('Current Mean Value:', mean_center_hsv)

            # mask the red color of the board
            mask = cv2.inRange(hsv, MIN_RED_HSV, MAX_RED_HSV)
            blanks = np.ones_like(color_image) * 255

            # convert to white and black frame
            black_out = cv2.bitwise_and(blanks, blanks, mask=mask)
            black_white = cv2.cvtColor(black_out, cv2.COLOR_BGR2GRAY)

            # contours
            contours, hierarchy = cv2.findContours(black_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            try:
                contour_idx = np.argmax([cv2.contourArea(cont) for cont in contours])

                cv2.drawContours(color_image, contours, contour_idx, (255, 255, 255), 3)
            except ValueError:
                pass
            # cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Color", color_image)
            # cv2.imshow("Depth", depth_image)
            keystroke = cv2.waitKey(1)
            if keystroke == ord('q'):
                break
            if keystroke == ord(" "):
                print("TOOK A SNAPSHOT")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    video_capture()