import os

from PIL import Image

from src.utils.Utils import download_images, images_path


def get_template(board_type, name, description):
    folder = images_path + name + "_" + description
    if  not os.path.exists(folder):
        os.makedirs(folder)
    if  not os.listdir(folder):
        print(f"No images for {board_type} board {name} with description {description}, downloading them now ...")
        download_images(board_type, name, description)
        print("Images downloaded successfully")
    image_path = os.path.join(folder, os.listdir(folder)[0])
    image_merge = Image.open(image_path).convert("RGBA")
    for file in os.listdir(folder):
        image_path = os.path.join(folder, file)
        image_merge = Image.alpha_composite(image_merge, Image.open(image_path).convert("RGBA"))
    return image_merge
