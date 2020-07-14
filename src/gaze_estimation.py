'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IECore
import cv2
import numpy as np
import logging as log



class GazeEstimator:
    '''
    Class for the Gaze Estimation Model.
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
        print('Input name ' + self.input_name)
        print("Model initialized")

    #def load_model(self):
    def load_model(self): 
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''

        # Initialize the plugin
        self.core = IECore()

        # Read the IR as a IENetwork
        self.exec_net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        print('GazeEstimatorNetwork loaded...')

        # Get the input layer
        self.input_blob = next(iter(self.exec_net.inputs))
        self.output_blob = next(iter(self.exec_net.outputs))
        print(self.input_blob)
        return

    def predict(self, head_pose, left_eye_crop, right_eye_crop):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_eye = self.preprocess_input(left_eye_crop)
        right_eye = self.preprocess_input(right_eye_crop)

        '''
        Makes an asynchronous inference request, given an input image.
        '''


        input_dict = {'head_pose_angles': head_pose, 'left_eye_image': left_eye, 'right_eye_image': right_eye}

        result = self.exec_net.infer(input_dict)



        vector = self.preprocess_output(result)
        #print(vector)


        return vector 


    def check_model(self):
        '''
        TODO: Check if this implementation is working with self.plugin...
        '''
        ### TODO: Check for supported layers ###
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
 
        input_img=cv2.resize(input_img, (60, 60), interpolation = cv2.INTER_AREA)
    
        # Change image from HWC to CHW
        input_img = input_img.transpose((2, 0, 1))
        input_img = input_img.reshape(1, *input_img.shape)
        
        return input_img 
 

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

        '''
        vector = outputs['gaze_vector'][0]

        return vector
