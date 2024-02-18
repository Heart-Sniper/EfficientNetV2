import os
import shutil

def collect_mistaken_images(mistaken_file_path: str, mistaken_img_collection_path: str) -> None:
    """
    Collects images listed in a file to a specified directory.
    
    Args:
    mistaken_file_path (str): Path to the file containing paths of mistaken images.
    mistaken_img_collection_path (str): Directory path where the mistaken images will be collected.
    
    Returns:
    None; this function copies files and prints relevant information.
    """
    # Create the directory if it does not exist
    if not os.path.exists(mistaken_img_collection_path):
        try:
            os.makedirs(mistaken_img_collection_path)
            print("Folder created.")
        except:
            print("Error creating folder.")
    else:
        print(f"Folder {mistaken_img_collection_path} already exists.")

    # Read the list of mistaken images
    with open(mistaken_file_path, "r") as file:
        mistaken_list = file.readlines()

    mistaken_img_path_list = [line.split(" ")[0] for line in mistaken_list]
    print(f"Number of mistaken images: {len(mistaken_img_path_list)}")

    # Copy the mistaken images to the collection path
    for img_path in mistaken_img_path_list:
        img_name = os.path.basename(img_path)
        aim_path = os.path.join(mistaken_img_collection_path, img_name)
        if os.path.exists(aim_path):
            print(f"Already exist: {img_name}")
        else:
            shutil.copy2(img_path, aim_path)
