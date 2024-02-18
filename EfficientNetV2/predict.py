import os
import csv
import json
import argparse

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import model as model


def parse_args():
    parser = argparse.ArgumentParser(description='Model Validation')

    parser.add_argument('--model-size', type=str, default="s")
    parser.add_argument("--model-path", type=str, help="Path to the model")
    parser.add_argument("--image-path", type=str, help="Path to images dataset, or single image")
    parser.add_argument("--class-json", type=int, default=2, help="Path to class json file")
    parser.add_argument("--save-path", type=str, 
                        help="Path to save txt file of misdetected image paths")

    return parser.parse_args()


def call_model_function(model_name, suffix):
    function_name = f"{model_name}_{suffix}"
    try:
        # Dynamically get the function based on the function_name
        model_function = getattr(model, function_name)
        # Call the function
        model_function()
    except AttributeError:
        print(f"Function {function_name} does not exist in the models module.")
        

def main(args):
    assert args.model_path is not None, "No model specified, please set --model-size"
    assert os.path.exists(args.img_path), "file: '{}' dose not exist.".format(img_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"torch device: {device}")

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "l"

    data_transform = transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                         transforms.CenterCrop(img_size[num_model][1]),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    img_path =args.img_path
        
    if os.path.isdir(img_path): 
        image_list = []
        image_name_list = os.listdir(img_path)
        for image_name in tqdm(image_name_list):
            image_list.append(os.path.join(img_path, image_name))
        print(f"images count: {len(image_list)}")
    elif os.path.isfile(img_path):
        image_list.append(img_path)
        print(f"image:\n{image_list}")

    json_path = args.class_json
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    create_model = call_model_function("efficientnetv2_", args.model_size)
    model = create_model(num_classes=2).to(device)

    model_weight_path = args.model_path
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()

    result = []
    for img in tqdm(image_list):
        img = Image.open(img).convert("RGB")
        img_tensor = data_transform(img)

        # Forward pass through the model
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

            result.append((img, class_indict[str(predict_cla)], predict[predict_cla].numpy()))
    
    # Specify the CSV file path
    csv_file_path = os.path.join(args.save_path, 'prediction_results.csv')

    # Writing to CSV
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Image Name', 'Predicted Class', 'Confidence'])

        for img_path, predicted_class, confidence in result:
            # Extract image name from the img_path
            img_name = os.path.basename(img_path)
            writer.writerow([img_name, predicted_class, confidence])

    print(f"Results saved to {csv_file_path}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
