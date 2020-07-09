import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
from argparse import ArgumentParser
from openvino.inference_engine import IECore  # used to load the IE python API

from input_feeder import InputFeeder
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
        help = "Path to .xml file of the face detection model")

    parser.add_argument("-hp", "--headposemodel",
        required = True,
        type = str,
        help = "Path to .xml file of the head pose estimation model")

    parser.add_argument("-fl", "--faciallandmarksmodel",
        required = True,
        type = str,
        help = "Path to .xml file of the facial landmark model")

    parser.add_argument("-ge", "--gazeestimationmodel",
        required = True,
        type = str,
        help = "Path to .xml file of the gaze estimation model")


    parser.add_argument("-i", "--input",
        required = True,
        type = str,
        help = "Path to video file or enter 'CAM' for webcam")

    parser.add_argument("-d", "--device",
        required = False,
        default = "CPU",
        type = str,
        help="Specify the target device to infer on: (CPU by default)")

    parser.add_argument("-l", "--cpu_extension",
        required = False,
        type = str,
        help = "Path to CPU extension if any layers are unsupported with choosen hardware")

    parser.add_argument("-p", "--probability_threshold",
        required = False,
        type = float,
        default = 0.6,
        help = "Probability threshold for model to identify the face, default = 0.6")   

    parser.add_argument("-vf", "--visual_flag",
        required = False,
        type = str,
        default = 0,
        help = "Flag for visualizing the outputs of the intermediate models, default = 0")     

    args = parser.parse_args()

    return args



    

def main():
    args = get_args()

    inputFile = args.input
    #inputFile = "./bin/demo.mp4"

    frame_count = 0

    focal_length = 950.0
    scale = 50

    if inputFile.lower() == "cam":
        feed = InputFeeder('cam')

    else:
        if not os.path.isfile(inputFile):
            log.error("Unable to find file: "+ inputFile)
            exit(1)
        feed = InputFeeder("video", inputFile)
        log.info("InputFeeder initialized")
    
    #print(args.facedetectionmodel)
    # Create instances of the different models
    fdm = FaceDetector(args.facedetectionmodel, args.device, args.cpu_extension)
    fdm.load_model()

    hpm = HeadPoseEstimator(args.headposemodel, args.device, args.cpu_extension)
    hpm.load_model()

    flm = FacialLandmarksDetector(args.faciallandmarksmodel, args.device, args.cpu_extension)
    flm.load_model()

    gem = GazeEstimator(args.gazeestimationmodel, args.device, args.cpu_extension)
    gem.load_model()


    feed.load_data()
    for ret, frame in feed.next_batch():
        if not ret:
            break
        
        if frame is not None:
            frame_count += 1

            key = cv2.waitKey(60)
            face_crop, face_coords = fdm.predict(frame.copy())
            #print(face_coords)
            print("Face crop shape: " + str(face_crop.shape))
            frame_h, frame_w = frame.shape[:2]

            #bounding_box = face_coords * np.array([frame_w, frame_h, frame_w, frame_h])
            #print(bounding_box)
            (xmin, ymin, xmax, ymax) = face_coords
            face_frame = frame[ymin:ymax, xmin:xmax]
            center_of_face = (xmin + face_frame.shape[1] / 2, ymin + face_frame.shape[0] / 2, 0) # 0 for colour channel
            print("Center of face " + str(center_of_face))
            
            #try:
            # Check if face was detected
            if type(face_crop) == int:
                print("Unable to detect face")
                if key == 27:
                    break
                continue

            left_eye_crop, right_eye_crop, landmarks = flm.predict(frame.copy())
            print("Landmarks" +str(landmarks))
            #print("Face crop shape: " + str(face_crop.shape))

            #print("Head pose trial")
            head_pose = hpm.predict(face_crop.copy())
            print("Head pose: " + str(head_pose))
            (pitch, roll, yaw)= head_pose


            
            # Send inputs to GazeEstimator
            gaze_vector = gem.predict(head_pose, left_eye_crop, right_eye_crop)
            print(gaze_vector)

            # visualize the axes of the HeadPoseEstimator results
            if args.visual_flag == 1:
                hdm.draw_axes(frame.copy(), center_of_face, yaw, pitch, roll, scale, focal_length)
            #except Exception as e:
            #    print("Unable to predict using model" + str(e) + " for frame " + str(frame_count))
            #continue

            
            


    cv2.destroyAllWindows()
    feed.close()



if __name__ == "__main__":
    main()