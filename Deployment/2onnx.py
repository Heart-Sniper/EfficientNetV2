import torch.onnx

import argparse
import onnx
import onnxruntime as ort

import EfficientNetV2.model as model

def parse_args():
    parser = argparse.ArgumentParser(description='Model Deployment')

    parser.add_argument('--class-num', type=int, default=2)
    parser.add_argument('--model-size', type=str, default="s",
                        help='Model size. Choosen from s, m or l')
    parser.add_argument('--model-path', type=str, 
                        help='Path to model which will be converted to ONNX format')
    parser.add_argument('--save-path', type=str,
                        help='Path to save the onnx model')
    
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
    # ensure device exists
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(f"torch device: {device}")
    print(f"onnx runtime device: {ort.get_device()}")

    img_size = {"s": [300, 384],  
                "m": [384, 480],
                "l": [384, 480]}
    img_resize = img_size[args.model_size]

    create_model = call_model_function("efficientnetv2_", args.model_size)
    model = create_model(num_classes=args.class_num).to(device)
    model.load_state_dict(torch.load(args.model_path))
    dummy_input = torch.randn(1, 3, img_resize[0], img_resize[1], requires_grad=True).cuda()

    torch.onnx.export(model, 
                    dummy_input, 
                    args.save_path, 
                    verbose=True, 
                    input_names = ['inputs'], 
                    output_names = ['outputs'],
                    do_constant_folding = True, )
    # check the converted ONNX model
    model = onnx.load(args.save_path)
    print(f"Model Check: {onnx.checker.check_model(model)}")


if __name__ == '__main__':
    args = parse_args()
    main(args)