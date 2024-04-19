from PIL import Image
import os
from pathlib import Path
from shutil import rmtree
from tqdm import tqdm

# set the path to the directory containing the images
list_dir = Path("./data/custom/self_dataset_sketch/").glob("*")
# print(f"[INFO] list_dir: {list(list_dir)}")
for cur_dir in tqdm(list(list_dir),desc="converting to jpg",total=len(list(list_dir))):
    # Set the input directory containing the images
    input_dir = str(cur_dir)
    print(f"[INFO] current directory: {input_dir}")

    # Set the output directory to save the converted images
    output_dir = input_dir

    # Iterate through each file in the input directory
    for filename in os.listdir(input_dir):
        # Check if the file is an image
        if filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".bmp") or filename.endswith(".gif") or filename.endswith(".JPG"):
            # Open the image file
            filepath = os.path.join(input_dir, filename)
            image = Image.open(filepath)
            image = image.convert('RGB')

            # Convert the image to JPEG and save it in the output directory
            new_filepath = os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpg")
            image.save(new_filepath, "JPEG")
            
            # remove the old file
            os.remove(filepath)
