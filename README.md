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

python3 src/main.py -i bin/demo.mp4 -fd models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -hp models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 -fl models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 -ge models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 -vf 1
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


## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
Limitations: Currently the model pipeline will only work with one detected face, all additional faces in the frame will be ignored.
<br>
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
