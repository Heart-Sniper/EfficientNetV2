import argparse
import torch
import torchvision.transforms as transforms

import model as model
from utils import read_split_data
import error_imgs as ei

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description='Model Validation')

    parser.add_argument('--model-size', type=str, default="s")
    parser.add_argument("--model-path", type=str, help="Path to the model")
    parser.add_argument("--dataset-path", type=str, help="Path to validation dataset")
    parser.add_argument("--num-class", type=int, default=2, help="Number of classes")
    parser.add_argument("--error-save-path", type=str, 
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

    __, __, val_images_path, val_images_label = read_split_data(args.dataset_path, val_rate=1.0)
    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    model_size = args.model_size
    data_transform = transforms.Compose([transforms.Resize(img_size[model_size]),
                                        transforms.CenterCrop(img_size[model_size]),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    val_dataset = ei.MyDataSet(images_path=val_images_path,
                               images_class=val_images_label,
                               transform=data_transform)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    
    create_model = call_model_function("efficientnetv2_", args.model_size)
    model = create_model(num_classes=args.num_class).to(device)
    model.load_state_dict(torch.load(args.model_path))
    __, __, classified_images_list = ei.evaluate(model=model,
                                                 data_loader=val_loader,
                                                 device=device,
                                                 epoch=1)
    
    if args.error_save_path is not None:
        txt = args.error_save_path + "/" + "mistaken_img_list.txt"
        with open(txt, "w") as file:
            for error_img_path in classified_images_list:
                file.write(f"{error_img_path[0]}  {error_img_path[1]} \n")


if __name__ == '__main__':
    args = parse_args()
    main(args)
