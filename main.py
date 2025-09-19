import ssl
import torch
import clip
from PIL import Image
import configparser
import sys
from pathlib import Path
import os
import shutil
from datetime import datetime
from tqdm import tqdm
import random


def select_10_same_prefix(input_dict):
    keys = list(input_dict.keys())
    random.shuffle(keys)
    shuffled_dict = {key: input_dict[key] for key in keys}
    groups = {}

    for key, value in shuffled_dict.items():
        if len(value) >= 2:
            prefix = tuple(value[:2])
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append((key, value))

    for prefix, items in groups.items():
        if len(items) >= 10:
            return dict(items[:10])

    return {}


def select_10_different_prefix(input_dict):
    keys = list(input_dict.keys())
    random.shuffle(keys)
    shuffled_dict = {key: input_dict[key] for key in keys}
    seen_prefixes = set()
    result = {}
    for key, value in shuffled_dict.items():
        if len(value) >= 2:
            prefix = tuple(value[:2])
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                result[key] = value
                if len(result) == 10:
                    break
    return result


if len(sys.argv) < 2:
    print("Usage: python script.py <absolute path to image folder> <mode>")
    sys.exit(1)

path_to_image_folder = sys.argv[1]
mode = sys.argv[2]


if mode not in ('similar', 'different'):
    print("not correct mode, try similar / different")
    sys.exit(1)

if not os.path.isdir(path_to_image_folder):
    print("Folder does not exist.")
    sys.exit(1)


config = configparser.ConfigParser()
config.read('config.cfg')
ssl_mode = config.getboolean('Settings', 'ssl_mode')


# для удобства скачиания - иначе блочится
if ssl_mode:
    ssl._create_default_https_context = ssl._create_unverified_context

device = "cuda" if torch.cuda.is_available() else "cpu"
with open('items.txt', 'r') as file:
    lines = file.readlines()

tags = []
for line in lines:
    tags.extend([item.strip() for item in line.split(',')])


tags =list(set(tags))

model, preprocess = clip.load("ViT-B/32", device=device)

text = clip.tokenize(tags).to(device)
image_extensions = {'.png', '.jpg', '.jpeg'}

# Поиск абсолютных путей к файлам из директории рекурсивный
image_paths = [
    str(p) for p in Path(path_to_image_folder).rglob('*')
    if p.suffix.lower() in image_extensions and p.is_file()
]

dict_helper = {}
for idx, img in tqdm(enumerate(image_paths)):
    # Для ускорения
    # if idx == 150:
    #     break
    image = preprocess(Image.open(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    dict_helper[img] = []
    for tag, prob in sorted(zip(tags, probs[0]), key=lambda x: x[1], reverse=True):
        # print(f"{tag}: {prob:.2%}")
        dict_helper[img].append(tag)

print("Done tagging")

current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
dir_name = f"files_{mode}_{time_string}"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

if mode=="similar":
    result = select_10_same_prefix(dict_helper)
else:
    result = select_10_different_prefix(dict_helper)
print(result)
for key, value in result.items():
    shutil.copyfile(key, f'{dir_name}/{key.split('/')[-1]}')
