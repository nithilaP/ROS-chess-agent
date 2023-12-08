import cv2
import cv2.aruco as aruco
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf2_ros
from sensor_msgs.msg import CameraInfo 
import matplotlib.pyplot as plt
import os
import time
import tf
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Header

rospy.init_node('sawyer_pick_place')

bridge = CvBridge()

#if using robot arm camera
def lookup_tag(tag_number):
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    rospy.sleep(3)
    try:
        trans = tf_buffer.lookup_transform('base', 'ar_marker_'+str(tag_number), rospy.Time())
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        print(e)
        print("Retrying ...")

    tag_pos = [getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')]
    return np.array(tag_pos)


#if using image msg from realsense
def detect_ar_marker(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        ar_marker_coords = corners[0] 
    else:
        return None

#from camera 
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])

dist_coeffs = np.array([k1, k2, p1, p2, k3])

marker_length = 0.1 #size of AR Tag 

def convert_to_camera_frame(ar_marker_coords):
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(ar_marker_corners, 
                                                    marker_length, camera_matrix, dist_coeffs)
    rmat = cv2.Rodrigues(rvec)[0]
    transform_matrix = np.zeros((4, 4))
    transform_matrix[:3, :3] = rmat
    transform_matrix[:3, 3] = tvec.T
    transform_matrix[3, 3] = 1
    point_camera_frame = np.dot(transform_matrix, ar_marker_coords)
    return point_camera_frame

    
def camera_callback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    ar_marker_coords = detect_ar_marker(cv_image)
    
    if ar_marker_coords is not None:
        camera_frame_coords = convert_to_camera_frame(ar_marker_coords)
    return camera_frame_coords

image_sub = rospy.Subscriber("/camera/color/image_raw", Image, camera_callback)

rospy.spin()
