'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IECore
import cv2

class FaceDetector:
    '''
    Class for the Face Detection Model.
    '''
    #def __init__(self, model_name, device='CPU', extensions=None):
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        print(model_name)

        try:
            self.model = IECore().read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?" + model_name)

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
        #result = result['detection_out']
        #result = np.squeeze(result)
        print(result)
        #print(result.shape)

        coordinates = self.preprocess_output(result)

        if (len(coordinates) == 0):
            return 0, 0


        coordinates = coordinates[0]    # only use the first returned image

        # Crop the face from the original image
        #[x_min, x_max, y_min, y_max]
        x_min = int(coordinates[0] * image.shape[1])
        x_max = int(coordinates[1] * image.shape[1])
        y_min = int(coordinates[2] * image.shape[0])
        y_max = int(coordinates[3] * image.shape[0])

        face_crop = image[y_min:y_max, x_min:x_max].copy()

        return coordinates, face_crop


    def check_model(self):
        '''
        TODO: Check if this implementation is working with self.plugin...
        '''
        ### TODO: Check for supported layers ###
        supported_layers = self.core.query_network(network=self.model,
                                                     device_name=self.device)
        unsupported_layers = []

        for l in self.network.layers.keys():
            if l not in supported_layers:
                unsupported_layers.append(l)
        
        if len(unsupported_layers) != 0:
            #print("Unsupported layers found: {}".format(unsupported_layers))
            #print("Check whether extensions are available to add to IECore.")
            exit(1)

        raise NotImplementedError

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
        #input_img = input_img.reshape(1, 3, h, w)
        input_img = input_img.reshape(n, c, h, w)
        #input_img = input_img.reshape(1, *input_img.shape)

        return input_img 

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        :param outputs: Frame which needs to be merged with the contours
        :return: coords values in form of list
        '''
        coords = []
        cnts = outputs[self.output_name][0][0]   
        for cnt in cnts: 
            conf = cnt[2]
            if conf > self.threshold:
                x_min = cnt[3]
                x_max = cnt[4]
                y_min = cnt[5]
                y_max = cnt[6]
                coords.append([x_min, x_max, y_min, y_max])
        return coords
