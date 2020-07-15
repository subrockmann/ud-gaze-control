import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import math

import logging as log
from argparse import ArgumentParser
from openvino.inference_engine import IECore  # used to load the IE python API
import pyautogui

from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import FaceDetector
from head_pose_estimation import HeadPoseEstimator
from facial_landmarks_detection import FacialLandmarksDetector
from gaze_estimation import GazeEstimator



def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = ArgumentParser("Run inference on an input video")

    parser.add_argument("-fd", "--facedetectionmodel",
        required = True,
        type = str,
        help = "Path to .xml file of the face detection model"
        )

    parser.add_argument("-hp", "--headposemodel",
        required = True,
        type = str,
        help = "Path to .xml file of the head pose estimation model"
        )

    parser.add_argument("-fl", "--faciallandmarksmodel",
        required = True,
        type = str,
        help = "Path to .xml file of the facial landmark model"
        )

    parser.add_argument("-ge", "--gazeestimationmodel",
        required = True,
        type = str,
        help = "Path to .xml file of the gaze estimation model"
        )

    parser.add_argument("-i", "--input",
        required = True,
        type = str,
        help = "Path to video file or enter 'CAM' for webcam"
        )

    parser.add_argument("-d", "--device",
        required = False,
        default = "CPU",
        type = str,
        help="Specify the target device to infer on: (CPU by default)"
            "Options are: CPU, GPU, FPGA and MYRIAD"
        )

    parser.add_argument("-l", "--cpu_extension",
        required = False,
        type = str,
        help = "Path to CPU extension if any layers are unsupported with choosen hardware"
        )

    parser.add_argument("-p", "--probability_threshold",
        required = False,
        type = float,
        default = 0.6,
        help = "Probability threshold for model to identify the face, default = 0.6"
        )   

    parser.add_argument("-vf", "--visual_flag",
        required = False,
        type = int,
        default = 0,
        help = "Flag for visualizing the outputs of the intermediate models, default = 0"
        )    

    parser.add_argument("-s", "--stats",
        required = False,
        type = str,
        default = 0,
        help = "Provide performance statistics, default = 0"
        )

    args = parser.parse_args()
    return args

def draw_bounding_box(frame, coords, colour=(0, 255, 0)):
    '''
    Draw bounding box
    '''
    viz_frame = frame.copy()
    cv2.rectangle(viz_frame, (coords[0], coords[1]), (coords[2], coords[3]),
                      colour, 2)
    return viz_frame

def visualize_landmark(frame, landmark, color = (255, 0, 0) ):
    '''
    Draw circle at landmark
    '''
    radius = 5
    thickness = 2
    viz_frame = frame.copy()
    cv2.circle(viz_frame, landmark, radius, color, thickness) 
    return viz_frame


def display_head_pose(frame, pitch, roll, yaw):
    '''
    Visualize head pose as displayed text
    '''
    viz_frame = frame.copy()
    cv2.putText(viz_frame,
                "head pose:",
                (20, 40),
                cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 1)
    cv2.putText(viz_frame,
            "pitch= {:.2f} , roll= {:.2f} , yaw= {:.2f}".format(
            pitch, roll, yaw),
            (20, 80),
            cv2.FONT_HERSHEY_COMPLEX,
            1, (0, 255, 0), 1)
    return viz_frame

def display_gaze(frame, gaze_vector):
    viz_frame = frame.copy()
    cv2.putText(viz_frame,
                "gaze:",
                (20, 120),
                cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 1)
    cv2.putText(viz_frame,
            "x= {:.2f} , y= {:.2f} , z= {:.2f}".format(
            gaze_vector[0], gaze_vector[1], gaze_vector[2]),
            (20, 160),
            cv2.FONT_HERSHEY_COMPLEX,
            1, (0, 255, 0), 1)
    return viz_frame

def visualize_gaze(frame, gaze_vector, landmarks):
            left_eye = (landmarks[0],landmarks[1])
            right_eye = (landmarks[2],landmarks[3])
            viz_frame = frame.copy()
            x, y = gaze_vector[:2]
            #head_frame = self.draw_head_pose()
            viz_frame = cv2.arrowedLine(viz_frame, left_eye, (int(left_eye[0]+x*200), int(left_eye[1]-y*200)), (0, 120, 20), 2)
            viz_frame = cv2.arrowedLine(viz_frame, right_eye, (int(right_eye[0]+x*200), int(right_eye[1]-y*200)), (0, 120, 20), 2)
            return viz_frame


def get_mouse_vector(gaze_vector, roll):
        """
        Create the vector for the mouse movement
        """
        cosv = math.cos(roll * math.pi / 180.0)
        sinv = math.sin(roll * math.pi / 180.0)

        newx = gaze_vector[0] * cosv + gaze_vector[1] * sinv
        newy = -gaze_vector[0] * sinv + gaze_vector[1] * cosv
        return newx, newy

    
def main():
    args = get_args()

    log.basicConfig(filename='example.log',level=log.DEBUG)

    inputFile = args.input
    #inputFile = "./bin/demo.mp4"

    mouse = MouseController("high", "fast")

    frame_count = 0
    focal_length = 950.0
    scale = 50

    #print(f"Visual flag: {args.visual_flag}")

    if inputFile.lower() == "cam":
        feed = InputFeeder('cam')
        log.info("Video source: "+ str(inputFile))

    else:
        if not os.path.isfile(inputFile):
            log.error("Unable to find file: "+ inputFile)
            exit(1)
        feed = InputFeeder("video", inputFile)
        log.info("Video source: "+ str(inputFile))
        log.info("InputFeeder initialized")
    
    log.info("Device: " + str(args.device))
    log.info("Face detection model: " + str(args.facedetectionmodel))
    log.info("Facial landmarks model: " + str(args.faciallandmarksmodel))
    log.info("Head pose estimation model: " + str(args.headposemodel))
    log.info("Gaze estimation model: " + str(args.gazeestimationmodel))

    if args.stats== 1:
        print("Running statistics...")
        inference_times = []
        start_time = time.time()

    # Create instances of the different models
    fdm = FaceDetector(args.facedetectionmodel, args.device, args.cpu_extension)
    fdm.load_model()
    fdm.check_model()

    hpm = HeadPoseEstimator(args.headposemodel, args.device, args.cpu_extension)
    hpm.load_model()
    hpm.check_model()

    flm = FacialLandmarksDetector(args.faciallandmarksmodel, args.device, args.cpu_extension)
    flm.load_model()
    flm.check_model()

    gem = GazeEstimator(args.gazeestimationmodel, args.device, args.cpu_extension)
    gem.load_model()
    gem.check_model()

    if args.stats== 1:
        duration_loading = time.time() - start_time
        print(f"Duration for loading and checking the models: {duration_loading}")
        log.info(f"Duration for loading and checking the models: {duration_loading}")

    cv2.namedWindow('preview',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('preview', 600,600)

    feed.load_data()
    for ret, frame in feed.next_batch():
        if not ret:
            break
        
        if frame is not None:
            frame_count += 1
            key = cv2.waitKey(60)

        if args.stats== 1:
            start_time = time.time()

            # Run face detection
            face_crop, face_coords = fdm.predict(frame.copy())
            #print("Face crop shape: " + str(face_crop.shape))
            frame_h, frame_w = frame.shape[:2]
            (xmin, ymin, xmax, ymax) = face_coords
            face_frame = frame[ymin:ymax, xmin:xmax]
            #center_of_face = (xmin + face_frame.shape[1] / 2, ymin + face_frame.shape[0] / 2, 0) # 0 for colour channel
            #print("Center of face " + str(center_of_face))
            
            try:
                # Check if face was detected
                if type(face_coords)== int:
                    print("Unable to detect face")
                    if key == 27:
                        break
                    continue

                # Facial landmark detection
                left_eye_crop, right_eye_crop, landmarks, crop_coords = flm.predict(face_crop.copy())
                #print("Landmarks" +str(landmarks))
                left_eye = (landmarks[0],landmarks[1])
                right_eye = (landmarks[2],landmarks[3])

                # Landmark position based on complete frame
                landmarks_viz = landmarks
                landmarks_viz[0] = landmarks_viz[0] + xmin
                landmarks_viz[1] = landmarks_viz[1] + ymin
                landmarks_viz[2] = landmarks_viz[2] + xmin
                landmarks_viz[3] = landmarks_viz[3] + ymin

                crop_coords_viz = (crop_coords[0] + xmin,
                    crop_coords[1] + ymin,
                    crop_coords[2] + xmin,
                    crop_coords[3] + ymin,
                    crop_coords[4] + xmin,
                    crop_coords[5] + ymin,
                    crop_coords[6] + xmin,
                    crop_coords[7] + ymin)

                left_eye_viz = (landmarks_viz[0],landmarks_viz[1])
                right_eye_viz = (landmarks_viz[2],landmarks_viz[3])

                third_eye_viz_x = (landmarks_viz[2]-landmarks_viz[0])/2 + landmarks_viz[0]
                third_eye_viz_y = (landmarks_viz[3]-landmarks_viz[1])/2 + landmarks_viz[1]
                third_eye_viz = (third_eye_viz_x, third_eye_viz_y)
                #print(landmarks_viz[0], landmarks_viz[2], third_eye_viz_x)

                # Head pose estimation
                head_pose = hpm.predict(face_crop.copy())
                print("Head pose: " + str(head_pose))
                (yaw, pitch, roll)= head_pose
                frame = display_head_pose(frame, pitch, roll, yaw)
                
                # Send inputs to GazeEstimator
                gaze_vector = gem.predict(head_pose, left_eye_crop, right_eye_crop)

                if args.stats== 1:
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)

                print(gaze_vector)
                frame = display_gaze(frame, gaze_vector)

                # Control the mouse
                if frame_count % 5 == 0:
                    mouse_x, mouse_y = get_mouse_vector(gaze_vector, roll)
                    print("Mouse vector:" + str(mouse_x) + " - " + str(mouse_y))
                    mouse.move(mouse_x, mouse_y)
                    currentMouseX, currentMouseY = pyautogui.position()
                    print("Mouse coordinates: " + str(currentMouseX)+ ", " + str(currentMouseY))

                if args.visual_flag == 1:

                    frame = draw_bounding_box(frame, face_coords)

                    left_eye_frame = crop_coords_viz[0:4]
                    right_eye_frame = crop_coords_viz[4:]
                    frame = draw_bounding_box(frame, left_eye_frame)
                    frame = draw_bounding_box(frame, right_eye_frame)

                    frame = visualize_landmark(frame, left_eye_viz)
                    frame = visualize_landmark(frame, right_eye_viz, color = (0, 0, 255) )

                    frame = visualize_gaze(frame, gaze_vector, landmarks_viz)

                    # visualize the axes of the HeadPoseEstimator results
                    #frame = hpm.draw_axes(frame.copy(), center_of_face, yaw, pitch, roll, scale, focal_length)
                    frame = hpm.draw_axes(frame.copy(), third_eye_viz, yaw, pitch, roll, scale, focal_length)
                    #hdm.draw_axes(frame.copy(), center_of_face, yaw, pitch, roll, scale, focal_length)

                cv2.imshow('preview', frame)
                cv2.imshow('left eye', left_eye_crop)
                cv2.imshow('right eye', right_eye_crop)

            except Exception as e:
                print("Unable to predict using model" + str(e) + " for frame " + str(frame_count))
                log.error("Unable to predict using model" + str(e) + " for frame " + str(frame_count))
            continue

    if args.stats== 1:
        avg_inference_time = sum(inference_times)/len(inference_times)
        log.info("Average inference time: " + str(avg_inference_time))
    cv2.destroyAllWindows()
    feed.close()

if __name__ == "__main__":
    main()