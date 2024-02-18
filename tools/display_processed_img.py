import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def process_and_display_image(image_path, resize_shape):
    """
    Process and display an image using specified transformations.
    
    Args:
    image_path (str): Path to the image file.
    resize_shape (list): The desired size for resizing the image, [width, height].
    
    Returns:
    None; this function displays the processed image.
    """
    # Load and convert the image
    img = Image.open(image_path).convert("RGB")

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transformations
    transform_img = transform(img)

    # Convert to numpy array and adjust for display
    image_np = transform_img.permute(1, 2, 0).numpy()
    image_np = (image_np * 0.5) + 0.5  # Adjust brightness and contrast
    image_np = image_np.clip(0, 1)    # Clip values to valid range

    # Display the image
    plt.imshow(image_np)
    plt.axis('off')  # Hide axes
    plt.show()
