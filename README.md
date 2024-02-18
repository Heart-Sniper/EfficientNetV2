# EfficientNet For Defect Detection

## Project Structue

```
|-- EfficientNetV2
    |-- Datasets
    |-- Deployment
    |-- EfficientNetV2
        |-- pretrained_model
        |-- runs
    |-- outputs
        |-- model
    |-- Results
    |-- Tools
    |-- README.md
```

## Data Preparation

```
|-- Datasets
    |-- dataset_name
        |-- class_name_0
            |-- 01.png
            |-- 02.png
            |-- ...
        |-- class_name_1
            |-- 01.png
            |-- 02.png
            |-- ...
        |-- label.txt
```

The label txt file format to follow:
```
image_name.png label
```

## Training

```shell
%cd EfficientNetV2
!torchrun EfficientNetV2/train.py \
        --model-size s \
        --weights EfficientNetV2/pretrained_model/pre_efficientnetv2-s.pth \ 
        --dataset-path Datasets/dataset_name \
        --num_classes <number_of_classes> \
        --epochs <number_of_epochs> \
        --batch-size 8 \
```

+ `--model-size` : Model category. Choose from "s", "m" or "l".
+ `--weights` : (Optional) Path to the pretrained model.
+ `--dataset-path` : Path to the dataset directory.
+ `--num_classes` : Number of categories to be classified.

## Validation

```shell
%cd EfficientNetV2
!torchrun EfficientNetV2/validation.py \
        --model-size s \
        --model-path <path_to_model> \
        --dataset-path <path_to_dataset> \
        --num-class <number_of_classes> \
        --error-save-path <path_to_save_error_result>
```

+ `--model-size` : Model category. Choose from "s", "m" or "l".
+ `--dataset-path` : Path to the dataset directory.
+ `--num_classes` : Number of categories to be classified.
+ `--error-save-path` : (Optional) If this parameter is set, the script will save the misdetected image path and result to the specified path.

## Prediction

Predictions for a single image or dataset based on specified model weights. Results will be saved as a .csv file to the specified path.

```shell
%cd EfficientNetV2
!torchrun EfficientNetV2/predict.py \
        --model-size s \
        --model-path <path_to_model> \
        --image-path <path_to_image> \
        --class-json <path_to_class_json_file> \
        --save-path <path_to_save_result_csv>
```

+ `--model-size` : Model category. Choose from "s", "m" or "l".
+ `--image-path` : Path to the single image or a directory of images to be predicted.
+ `--class-json` : Path to JSON file of category and corresponding number. </br>
    Format folows:
    ```JSON
    {
      "0": "category0",
      "1": "category1",
      ...
    }
    ```
+ `--save-path` : Path to save the resulting .csv file.
  It will be saved as the format like this: </br>
  ```
  | image path | predicted class | confidence |
  ```

## Deployment (to ONNX)

```shell
%cd EfficientNetV2
! torchrun Deployment/2onnx.py \
    --class-num 2 \
    --model-size s \
    --model-path <path_to_pth_model> \
    --save-path <save_path>
```

+ `--class-num` : Number of classes
+ `--model-size` : Model category. Choose from "s", "m" or "l".