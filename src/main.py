import os
import sys
import time
import socket
import json
import cv2

import logging as log
from argparse import ArgumentParser
from openvino.inference_engine import IECore  # used to load the IE python API

from input_feeder import InputFeeder
from face_detection import FaceDetector


def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = ArgumentParser("Run inference on an input video")


    parser.add_argument("-fd", "--facedetectionmodel",
        required = True,
        type = str,
        help = "Path to .xml file of the face detection model")

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

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    inputFile = args.input
    #inputFile = "./bin/demo.mp4"

    frame_count = 0

    if inputFile.lower() == "cam":
        feed = InputFeeder('cam')

    else:
        if not os.path.isfile(inputFile):
            print("Unable to find file: "+ inputFile)
            exit(1)
        feed = InputFeeder("video", inputFile)
        
    
    print(args.facedetectionmodel)
    # Create instances of the different models
    fdm = FaceDetector(args.facedetectionmodel, args.device, args.cpu_extension)
    fdm.load_model()



    feed.load_data()
    for ret, frame in feed.next_batch():
        if not ret:
            break
        
        if frame is not None:
            frame_count += 1

            key = cv2.waitKey(60)
            face_crop, face_coords, = fdm.predict(frame.copy())

            print(face_coords)


    cv2.destroyAllWindows()
    feed.close()



if __name__ == "__main__":
    main()