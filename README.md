# Individual-project
# Developing a deep learning model to differentiate between military and civilian vehicles
## Overview
The project trains and evaluates multiple CNN architectures on the Militairy and Civlian Vehicles CLassification Dataset (see below). The project also uses a trained architecture with YOLOv8n to detect and classify civilian and militairy vehicles in real-time.

## Dataset
This project uses the Military and Civilian Vehicles Classification Dataset, which is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

Dataset link: https://data.mendeley.com/datasets/njdjkbxdpn/1? License: https://creativecommons.org/licenses/by/4.0/

For the detection pipeline, videos sourced from [pexels](https://www.pexels.com) were used.

I acknowledge the original authors of the dataset. This dataset was used for the purpose of training and evaluating machine learning models.

## Project Structure
* config.py: changing of hyperparamters and path management.
* dataset.py: load and preprocess data
* train.py: training pipeline
* evaluate.py: evaluation pipeline
* model.py: Define model architectures
* detect.py: Real-time detection pipeline
* utils.py: helper files

## Usage
Follows this pipeline: train model -> evaluate model/use model for detection
1. train a model: python train.py --model [simple/resnet50/efficientnet] --no_class_weights (to turn off class weights - on by default)
2. evaulate a model: python evaluate.py --model [simple/resnet50/efficientnet] (only works if the chosen model has already been trained)
3. run detection: python detect.py --video path/to_video.mp4 --model [simple/resnet50/efficientnet] --yolo_conf x --cls_conf y (for some floats 0 < x,y < 1)

To freeze or unfreeze a models backbone, open model.py and find the get_model() function. Within this function change freeze_backbone to true/false for the model you want to freeze/unfreeze.

Due to the size of checkpoint files, they cannot be included in the repository and therefore the entire pipeline must be followed if the user wishes to run a detection. 

The tank video used in this project is also to large to be included in the repository, but it can be downloaded by following this links. Please copy the file into the videos folder of the repository.
* Tank Video: https://www.pexels.com/video/a-tank-with-a-large-amount-of-rusted-metal-on-it-18359147/
## Known Limitations
* YOLOv8n is not fine-tuned on military and civilian vehicles
* Vehicles outside the 6 classes in the dataset will be misclassified
* Detection pipeline is qualatative only and is not optimised for performance resulting in a low fps
* To end a detection video early the user must crtl+c in the command line
