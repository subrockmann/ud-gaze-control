# Computer Pointer Controller

*TODO:* Write a short introduction to your project

## Project Set Up and Installation

pip3 install -r requirements.txt

source /opt/intel/openvino/bin/setupvars.sh

## Required models 
### Face detection model
https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html

sudo ./downloader.py --name face-detection-adas-binary-0001 -o /models

### Head pose estimation model
https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html

sudo ./downloader.py --name head-pose-estimation-adas-0001 -o /models

### Facial landmark detection model
https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html

sudo ./downloader.py --name landmarks-regression-retail-0009 -o /models

### Gaze estimation model
https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html

sudo ./downloader.py --name gaze-estimation-adas-0002 -o /models





*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

## Demo
*TODO:* Explain how to run a basic demo of your model.
From inside the starter folder:
python3 src/main.py -i bin/demo.mp4 -fd models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -hp models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 -fl models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009


## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
