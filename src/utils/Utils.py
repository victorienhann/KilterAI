import collections
import io
import os
import re
import sqlite3
import zipfile
from pathlib import Path

import torch

import h5py
import pandas as pd
import requests
from PIL import ImageDraw, Image

from src.ai.generator.TokenVariationalAutoencoder import TokenVariationalAutoEncoder
from src.utils.Queries import QUERIES

# All resource paths are anchored to the repo root regardless of the current
# working directory, since different entry points (src/Main.py, MainVisu.py,
# tests, ...) are run from different places.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCES_DIR = PROJECT_ROOT / "resources"

HOST_BASES = {
    "aurora": "auroraboardapp",
    "decoy": "decoyboardapp",
    "grasshopper": "grasshopperboardapp",
    "kilter": "kilterboardapp",
    "soill": "soillboardapp",
    "tension": "tensionboardapp2",
    "touchstone": "touchstoneboardapp",
}

APP_PACKAGE_NAMES = {
    "aurora": "auroraboard",
    "decoy": "decoyboard",
    "grasshopper": "grasshopperboard",
    "kilter": "kilterboard",
    "soill": "soillboard",
    "tension": "tensionboard2",
    "touchstone": "touchstoneboard",
}

ROLES = {"start" : [12, 20, 24, 28, 32, 42],
         "middle" : [13, 21, 25, 29, 33, 43],
         "finish" : [14, 22, 26, 30, 34, 44],
         "foot" : [15, 23, 27, 31, 35, 45]}

images_path = str(RESOURCES_DIR / "images") + "/"

def download_images(board, name, description):
    """
    Download all images for a given board.

    :param board: The board type
    :param name: Board name
    :param description: Board description
    """
    output_directory = os.path.join(images_path, name + "_" + description)
    os.makedirs(output_directory, exist_ok=True)
    # As of mid-2026 Aurora Climbing's API dropped the "api." subdomain (trademark
    # dispute forced them off the kilterboardapp.com "api." host - see
    # https://www.climbing.com/news/why-the-kilter-board-app-suddenly-disappeared/).
    # api.{board}app.com is now NXDOMAIN; the bare {board}app.com host is current,
    # matching https://github.com/lemeryfertitta/BoardLib's WEB_HOSTS.
    api_host = f"https://{HOST_BASES[board]}.com"

    database_path = str(RESOURCES_DIR / "databases" / f"{board}.sqlite")
    connection = sqlite3.connect(database_path)
    res = pd.read_sql_query(QUERIES["images"], connection, None, None, {'name': name, 'description': description})
    connection.close()

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    for image_filename in res["image_filename"]:
        # Create subdirectories if needed (e.g., for product_sizes_layouts_sets/1-v4.png)
        image_filename_short = image_filename.split("/")[-1]
        output_path = os.path.join(output_directory, image_filename_short)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Skip download if file already exists
        if os.path.exists(output_path):
            print(f"Skipping {image_filename_short} (already exists)")
            continue

        response = requests.get(
            f"{api_host}/img/{image_filename}",
            headers=headers,
        )
        response.raise_for_status()

        with open(output_path, "wb") as output_file:
            output_file.write(response.content)

def download_database(board):
    """
    The sqlite3 database is stored in the assets folder of the APK files for the Android app of each board.

    This function downloads the latest APK file for the board's Android app and extracts the database from it.
    :param board: The board to download the database for.
    :param output_file: The file to write the database to.
    """
    app_package_name = APP_PACKAGE_NAMES[board]
    response = requests.get(
        f"https://d.apkpure.net/b/APK/com.auroraclimbing.{app_package_name}",
        params={"version": "latest"},
        # Some user-agent is required, 403 if not included
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        },
    )
    response.raise_for_status()
    output_file = str(RESOURCES_DIR / "databases" / f"{board}.sqlite")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    bundle_file = io.BytesIO(response.content)
    with zipfile.ZipFile(bundle_file, "r") as zip_file:
        try:
            apk_file = io.BytesIO(zip_file.read(f"com.auroraclimbing.{app_package_name}.apk"))
        except KeyError:
            # Fallback to old APK directory structure to support older versions
            with open(output_file, "wb") as output_file:
                output_file.write(zip_file.read("assets/db.sqlite3"))
        else:
            with zipfile.ZipFile(apk_file, "r") as main_zip:
                with open(output_file, "wb") as output_file:
                    output_file.write(main_zip.read("assets/db.sqlite3"))

def connect_to_database(board):
    database_path = RESOURCES_DIR / "databases" / f"{board}.sqlite"
    if not database_path.exists():
        print(f"Missing database for {board} board")
        print(f"Downloading database for {board} board ...")
        download_database(board)
        print("Database downloaded successfully.")
    return sqlite3.connect(str(database_path))

def extract_roles(climb):
    start = []
    middle = []
    finish = []
    foot = []

    pattern = r"p(\d+)r(\d+)"
    matches = re.findall(pattern, climb)
    for p, r in matches:
        if int(r) in ROLES["start"]:
            start.append(int(p))
        elif int(r) in ROLES["middle"]:
            middle.append(int(p))
        elif int(r) in ROLES["finish"]:
            finish.append(int(p))
        elif int(r) in ROLES["foot"]:
            foot.append(int(p))
        else :
            raise ValueError("Invalid role {}".format(r))
    return start, middle, finish, foot

def make_circle(color, size=50):
    """
    Draws a circle with center (x, y) on an RGBA image.

    Parameters:
        color (tuple): RGBA color of the circle
        size (tuple): Size of the output image (width, height)

    Returns:
        PIL.Image: Image with the circle drawn
    """
    circle = Image.new("RGBA", (size, size), (0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(circle)
    left_up = (0, 0)
    right_down = (size, size)
    draw.ellipse([left_up, right_down], outline=color, width=4)
    return circle

datasets_path = str(RESOURCES_DIR / "datasets")

def save_dataset(dataset, name, description):
    os.makedirs(datasets_path, exist_ok=True)
    filename = f"{datasets_path}/{name}_{description}.h5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("tokens", data=dataset.tokens)
        f.create_dataset("angles", data=dataset.angles)
        f.create_dataset("grades", data=dataset.grades)
        print(f"Dataset successfully exported to {filename}")

def load_dataset(name, description):
    filename = f"{datasets_path}/{name}_{description}.h5"
    return h5py.File(filename, "r")

models_path = str(RESOURCES_DIR / "models")

def save_model(model, name, description):
    os.makedirs(models_path, exist_ok=True)
    filename = f"{models_path}/{name}_{description}.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "vocab_size": model.vocab_size,
        "angles_min": model.angles_min,
        "angles_max": model.angles_max,
        "grades_min": model.grades_min,
        "grades_max": model.grades_max,
        "latent_dim": model.latent_dim,
    }, filename)
    print(f"Model successfully saved to {filename}")

def load_model(name, description):
    filename = f"{models_path}/{name}_{description}.pth"

    ckpt = torch.load(filename, weights_only=False)  # autorise la lecture complète
    model = TokenVariationalAutoEncoder(ckpt["vocab_size"], ckpt["angles_min"], ckpt["angles_max"], ckpt["grades_min"], ckpt["grades_max"], ckpt["latent_dim"])
    model.load_state_dict(ckpt["state_dict"])
    print(f"Model successfully loaded from {filename}")
    return model

def load_name_and_description(connection):
    df = pd.read_sql_query(QUERIES["name_description"], connection)
    dict = collections.defaultdict(list)
    for row in df.values:
        if row[0] not in dict.keys():
            dict[row[0]] = [row[1]]
        else :
            dict[row[0]].append(row[1])
    return dict




