'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IECore
import cv2
import numpy as np
import logging as log
import math



class HeadPoseEstimator:
    '''
    Class for the Head Pose Estimation Model.
    '''
    #def __init__(self, model_name, device='CPU', extensions=None):
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device

        print(model_name)

        try:
            self.model = IECore().read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?" + model_name)
            log.error("Head Pose Estimation initialization failed", e)
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        print('Output name ' + self.output_name)
        print("Model initialized")

    #def load_model(self):
    def load_model(self): #, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''

        # Initialize the plugin
        self.core = IECore()

        # Read the IR as a IENetwork
        self.exec_net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        print('Network loaded...')

        # Get the input layer
        self.input_blob = next(iter(self.exec_net.inputs))
        self.output_blob = next(iter(self.exec_net.outputs))
        #print(self.input_blob)
        return

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''

        p_frame = self.preprocess_input(image)

        '''
        Makes an asynchronous inference request, given an input image.
        '''
        #print(self.input_blob)
        input_name = self.input_name
        input_dict = {self.input_name: p_frame}

        result = self.exec_net.infer(input_dict)
        #print("Head pose output " + str(result))

        axes = self.preprocess_output(result)
        return axes


    def check_model(self):
        '''
        TODO: Check if this implementation is working with self.plugin...
        '''

        supported_layers = self.core.query_network(network=self.model,
                                                     device_name=self.device)
        unsupported_layers = []

        for l in self.model.layers.keys():
            if l not in supported_layers:
                unsupported_layers.append(l)
        
        if len(unsupported_layers) != 0:
            log.warning("Unsupported layers found: {}".format(unsupported_layers))
            log.warning("Check whether extensions are available to add to IECore.")
            #sys.exit("Add necessary extension for given hardware")
            #print("Unsupported layers found: {}".format(unsupported_layers))
            exit(1)

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        input_img = image
        
        # Preprocessing input
        n, c, h, w = self.input_shape


    
        input_img=cv2.resize(input_img, (w, h), interpolation = cv2.INTER_AREA)
    
        # Change image from HWC to CHW
        input_img = input_img.transpose((2, 0, 1))
    
        input_img = input_img.reshape(n, c, h, w)


        return input_img 

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

        '''
        pitch = np.squeeze(outputs['angle_p_fc'])
        roll = np.squeeze(outputs['angle_r_fc'])
        yaw = np.squeeze(outputs['angle_y_fc'])
        axes = np.array([yaw, pitch, roll])
        return axes


# Visualization of head pose estimation 
    # Code for draw_axes() and build_camera_matrix from https://knowledge.udacity.com/questions/171017
    def draw_axes(self, frame, center_of_face, yaw, pitch, roll, scale, focal_length):
        yaw *= np.pi / 180.0
        pitch *= np.pi / 180.0
        roll *= np.pi / 180.0
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(pitch), -math.sin(pitch)],
                       [0, math.sin(pitch), math.cos(pitch)]])
        Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                       [0, 1, 0],
                       [math.sin(yaw), 0, math.cos(yaw)]])
        Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                       [math.sin(roll), math.cos(roll), 0],
                       [0, 0, 1]])

        R = Rz @ Ry @ Rx
        camera_matrix = self.build_camera_matrix(center_of_face, focal_length)
        xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
        yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
        zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
        zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
        o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
        o[2] = camera_matrix[0][0]
        xaxis = np.dot(R, xaxis) + o
        yaxis = np.dot(R, yaxis) + o
        zaxis = np.dot(R, zaxis) + o
        zaxis1 = np.dot(R, zaxis1) + o
        xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
        xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
        xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
        yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
        p1 = (int(xp1), int(yp1))
        xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        cv2.circle(frame, p2, 3, (255, 0, 0), 2)
        return frame

    def build_camera_matrix(self, center_of_face, focal_length):
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        camera_matrix = np.zeros((3, 3), dtype='float32')
        camera_matrix[0][0] = focal_length
        camera_matrix[0][2] = cx
        camera_matrix[1][1] = focal_length
        camera_matrix[1][2] = cy
        camera_matrix[2][2] = 1
        return camera_matrix        


    

