# Using Microsoft's FocalNet architecture to do object detection on arthropods

"""
This project focuses on improving classifications on the Arthropod Taxonomy Orders Object Detection Dataset (ArTaxOr) dataset from Kaggle. ArTaxOr contains 15,376 images of arthropods from 7 different orders. Its images may contain multiple arthropods of different orders within one scene. All arthropod objects are labelled and bounding box data is provided for each arthropod (https://www.kaggle.com/datasets/mistag/arthropod-taxonomy-orders-object-detection-dataset/data). The ArTaxOr dataset provides an open object-detection problem on which continuous improvements can be made.

The problem of object detection relating to arthropods is a useful one to contemplate. The phylum Arthropoda is the largest phylum in terms of both diversity and numbers in the animal kingdom (https://www.britannica.com/animal/arthropod). The phylum Arthropoda includes the subphylum Hexapoda, which contains the class Insecta, commonly known as insects. Insects have a large impact upon human quality of life in almost every aspect of human life. Insects can affect human crops, human health, pets, housing, and many other important aspects of life. In addition to the impact that insects have upon human well-being, insects have a large amount of novelty, entertainment, and educational value for people. I propose to apply an advanced deep learning architecture to this problem and create a robust application for arthropod localization and identification.
"""

# Load the necessary libraries
import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor
import os
from PIL import Image
import json
from glob import glob
from datasets import Dataset, DatasetDict
import random

# Set the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Strings for important variables
data_root_path = "ArTaxOr"
hf_model_string = "microsoft/focalnet-base"

def get_pair_dict(order_dir, annotation_data, filename_to_id, cat_to_id):
    """
    Given a directory for a given order, annotation data, and a filename-to-ID dictionary,
    return a dictionary that can be converted to a Hugging Face dataset.

    Args:
        order_dir (str): Directory path for the given order.
        annotation_data (dict): Annotation data for the image.
        filename_to_id (dict): Dictionary mapping filenames to unique IDs.
        cat_to_id (dict): Dictionary mapping category names to unique IDs.

    Returns:
        dict: A dictionary containing image metadata and object annotations.
    """
    # Get image file name and file path
    img_filename = annotation_data['asset']['name']
    img_filepath = os.path.join(order_dir, img_filename)

    # Open the image to get its dimensions
    img = Image.open(img_filepath)
    width, height = img.size

    # Initialize lists for objects
    objects = {
        'bbox': [],
        'category': [],
        'id': [],
        'area': []
    }

    # Initialize unique ID counter for objects
    id_count = 0

    # Process each region in the annotation data
    for region in annotation_data['regions']:
        # Get bounding box and convert to COCO format
        bb = region['boundingBox']
        bb_coco = [bb['left'], bb['top'], bb['width'], bb['height']]
        objects['bbox'].append(bb_coco)

        # Calculate area of the bounding box
        area = bb['width'] * bb['height']
        objects['area'].append(area)

        # Get category from tags and look up category ID in cat_to_id
        tags = [tag for tag in region['tags'] if "_" not in tag]
        category = tags[0] if tags else "unknown"
        category_id = cat_to_id.get(category, -1)  # Use -1 or another default value if category is not found
        objects['category'].append(category_id)

        # Assign a unique ID to each object
        id_count += 1
        objects['id'].append(id_count)

    # Get or assign a unique image ID
    if img_filename in filename_to_id:
        image_id = filename_to_id[img_filename]
    else:
        image_id = len(filename_to_id) + 1
        filename_to_id[img_filename] = image_id

    # Create dictionary with image and additional metadata
    pair_dict = {
        'height': height,
        'width': width,
        'image': img,
        'image_id': image_id,
        'objects': objects
    }

    return pair_dict

filename_to_id = {}
cat_to_id = {
    "Araneae": 1,
    "Coleoptera": 2,
    "Diptera": 3,
    "Hemiptera": 4,
    "Hymenoptera": 5,
    "Lepidoptera": 6,
    "Odonata": 7
}

# This provides mappings for file ids and category ids to their respective names
id_to_filename = {v: k for k, v in filename_to_id.items()}
id_to_cat = {v: k for k, v in cat_to_id.items()}

def generate_dataset(dir, filename_to_id, cat_to_id):
    '''
    Generator that yields text-image pair dictionaries from the file structure.
    '''
    for order in ["Araneae", "Coleoptera", "Diptera", "Hemiptera", "Hymenoptera", "Lepidoptera", "Odonata"]:
        order_dir = os.path.join(dir, order)
        annotations_dir = os.path.join(order_dir, "annotations")
        for annotation_filepath in glob(f"{annotations_dir}/*.json"):
            with open(annotation_filepath, 'r') as file:
                annotation_data = json.load(file)
            pair_dict = get_pair_dict(order_dir, annotation_data, filename_to_id, cat_to_id)
            yield pair_dict

# Create the dataset from the generator
ds = Dataset.from_generator(generate_dataset, gen_kwargs={'dir': data_root_path, 'filename_to_id': filename_to_id, 'cat_to_id': cat_to_id})

# Define the split ratios
train_test_split = ds.train_test_split(test_size=0.2)
test_valid_split = train_test_split['test'].train_test_split(test_size=0.5)

# Create a DatasetDict
dd = DatasetDict({
    'train': train_test_split['train'],
    'test': test_valid_split['test'],
    'validation': test_valid_split['train']
})

# Save the dataset to disk
dd.save_to_disk("ArTaxOr_HF_dataset")
