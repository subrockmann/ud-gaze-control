# Computer Pointer Controller

This project is using 4 pre-trained computer vision models to control the mouse pointer
with the eye gaze. Possible inputs are a .mp4 file or the input from a webcam.
<br>


## Project Set Up and Installation

1. Install the [OpenVINO Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html) on your machine according to the detailed instructions.

2. Clone this repository from https://github.com/subrockmann/ud-gaze-control.git

3. Change into the gaze-control directory and source the local environment
```
cd gaze-control
source venv/bin/acitvate
```

5. Install the requirements
```
pip3 install -r requirements.txt
```

## Download the pre-trained models from the Open Model Zoo
### 1. Face detection model
https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html

```
sudo python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001 -o ./models
```

### 2. Head pose estimation model
https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html

```
sudo python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 -o ./models
```

### 3. Facial landmark detection model
https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html

```
sudo python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 -o ./models
```

### 4. Gaze estimation model
https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html

```
sudo python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 -o ./models
```

# Demo
Set the environment variables for openVINO
```
source /opt/intel/openvino/bin/setupvars.sh
```

From inside the gaze-control folder you can run the following listed commands according to their specification:

* Inference on video file located at bin/demo.mp4 using CPU and precision FP32
```
python3 src/main.py -i bin/demo.mp4 -fd models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -hp models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 -fl models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 -ge models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 -vf 1 -s 1
```

* Inference on video file located at bin/demo.mp4 using CPU and precision FP16
```
python3 src/main.py -i bin/demo.mp4 -fd models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -hp models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -fl models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 -ge models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 -vf 1 -s 1
```

* Inference on video file located at bin/demo.mp4 using CPU and precision FP16-INT8
```
python3 src/main.py -i bin/demo.mp4 -fd models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -hp models/intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001 -fl models/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009 -ge models/intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002 -vf 1 -s 1
```



## Documentation
Command line arguments

| argument    | type           | default |  description  |
| ------------- |:-------------:|:---:|-----:|
| -fd    | required | none | Path to .xml file of the face detection model |
| -hp    | required | none | Path to .xml file of the head pose estimation model|
| -fl   | required | none | Path to .xml file of the facial landmark model|
| -ge   | required | none | Path to .xml file of the gaze estimation model|
| -i    | required | none | Path to video file or enter 'CAM' for webcam|
| -d    | optional | CPU | Target for inference - options are: CPU, GPU, FPGA and MYRIAD|
| -p    | optional | 0.6 | Probability threshold for model to identify the face| 
| -vf    | optional | 0 | Flag for visualizing the outputs of the intermediate models|
| -s    | optional | 0 | Flag for providing performance statistics|


## File structure
```
    ./bin
        ./demo.mp4 - demo video for inference

    ./models/intel - models directory

    ./src/  - source code of the project

        ./facial_landmarks_detection.py - code for handling the facial landmark detection model

        ./head_pose_estimation.py -code for handling the head pose estimation model

        ./input_feeder.py - code to load inputs from video or camera 

        ./mouse_controller.py - code to move the mouse based on the output from the gaze estimation model

        ./face_detection.py  - code for handling the face detection model

        ./gaze_estimation.py - code for handling the gaze estimation model

        ./main.py - main file for running the project

    ./venv/ - folder that contains the virtual environment 

    ./example.log - log file for statistics and debugging

    .requirements.txt - contains python packages required for running the application

    ./README.md  - project description
```

## Benchmarks

### model sizes depending on model precision

**face-detection-adas-binary-0001**

| precision         | size of model |
|--------------|---------------|
|  FP32-INT1   |  1.86 MB        |

**head-pose-estimation-adas-0001**

| precision         | size of model |
|--------------|---------------|
|  FP16   |  3.69 MB       |
|  FP16-INT8   |  2.05 MB        |
|  FP32   |  7.34 MB       |

**landmarks-regression-retail-0009**

| precision         | size of model |
|--------------|---------------|
|  FP16   |  413 KB      |
|  FP16-INT8   | 314 KB        |
|  FP32   |  786 KB       |

**gaze-estimation-adas-0002**

| precision         | size of model |
|--------------|---------------|
|  FP16   |  3.65 MB       |
|  FP16-INT8   |  2.05 MB      |
|  FP32   |   7.24 MB       |

### model loading times depending on model precision

| model    | precision           | loading time in s | 
| ------------- |:-------------:|:-----:|
| face-detection-adas-binary-0001| FP32-INT1 | 0.1897 |
| head-pose-estimation-adas-0001| FP32 | 0.0811 |
| head-pose-estimation-adas-0001| FP16 | 0.1377 |
| head-pose-estimation-adas-0001| FP16-INT8 | 0.3664 |
| landmarks-regression-retail-0009| FP32 | 0.0836 |
| landmarks-regression-retail-0009| FP16 | 0.0896 |
| landmarks-regression-retail-0009| FP16-INT8 | 0.1266 |
| gaze-estimation-adas-0002| FP32 | 0.1276 |
| gaze-estimation-adas-0002| FP16 | 0.1672 |
| gaze-estimation-adas-0002| FP16-INT8 | 0.4030 |


## Results
I was surprised that the loading times for the smaller FP-16 and Fp16-INT8 models where higher than for the larger FP32 models. Apparently Intel CPUs are calculating on 32 bits and casting all the lower precision numbers to 32 bits. There is also no significant inference time difference (average inference time on all precisions is about 0.036 s). Therefore the different model precisions will only be important when running inference on other devices.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. 
<br>
Limitations: Currently the model pipeline will only work with one detected face, all additional faces in the frame will be ignored.
