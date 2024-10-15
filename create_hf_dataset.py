'''
Script motivation: create a coco-format dataset.

This script is specifically for getting data from the ArTaxOr dataset (script assumes it exists in the same directory as itself)
and processing it to coco format. Below I have listed the requirements that I have for a COCO format dataset:
1. Must have a .json manifest file
2. Must have a directory where all images are stored.

Manifest file requirements: 
A. It is a json file
B. It has the following categories: Info, Images, Annotations, Categories.

Basic structure: 
{
    'info': {'description': ..., 'version': ..., 'year': ..., 'date_created': ...},
    'images': [{'id': ..., 'width': ..., 'height': ..., 'date_created': ..., 'file_name': 'example.jpg'}, ...]
    'annotations': [{'id': ..., 'category_id': ..., 'image_id': ..., 'area': ..., 'bbox': [..., ..., ..., ...]}, ...],
    'categories': [{'id': ..., 'name': ...}, ...]
}

'''

import os
import json
from glob import glob
from PIL import Image
from datetime import datetime
import argparse

class HF_Dataset_Generator:

    def __init__(self, data_root_path):
        self.data_root_path = data_root_path
        self.filename_to_id = {}
        self.cat_to_id = {
            "Araneae": 1,
            "Coleoptera": 2,
            "Diptera": 3,
            "Hemiptera": 4,
            "Hymenoptera": 5,
            "Lepidoptera": 6,
            "Odonata": 7
        }
        
        self.id_to_filename = {v: k for k, v in self.filename_to_id.items()}
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}

    def get_pair_dict(self, order_dir, annotation_data):
        """
        Given a directory for a given order, annotation data, and a filename-to-ID dictionary,
        return a dictionary that can be converted to a Hugging Face dataset.

        Args:
            order_dir (str): Directory path for the given order.
            annotation_data (dict): Annotation data for the image.

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
            category_id = self.cat_to_id.get(category, -1)  # Use -1 or another default value if category is not found
            objects['category'].append(category_id)

            # Assign a unique ID to each object
            id_count += 1
            objects['id'].append(id_count)

        # Get or assign a unique image ID
        if img_filename in self.filename_to_id:
            image_id = self.filename_to_id[img_filename]
        else:
            image_id = len(self.filename_to_id) + 1
            self.filename_to_id[img_filename] = image_id

        # Create dictionary with image and additional metadata
        pair_dict = {
            'height': height,
            'width': width,
            'image': img,
            'image_id': image_id,
            'objects': objects,
            'file_name': img_filename
        }

        return pair_dict

    def generate_dataset_coco(self):
        '''
        Generator that yields text-image pair dictionaries from the file structure.
        '''
        for order in ["Araneae", "Coleoptera", "Diptera", "Hemiptera", "Hymenoptera", "Lepidoptera", "Odonata"]:
            order_dir = os.path.join(self.data_root_path, order)
            annotations_dir = os.path.join(order_dir, "annotations")
            for annotation_filepath in glob(f"{annotations_dir}/*.json"):
                with open(annotation_filepath, 'r') as file:
                    annotation_data = json.load(file)
                pair_dict = self.get_pair_dict(order_dir, annotation_data)
                yield pair_dict

    def create_coco_manifest(self, output_path):
        '''
        Create a COCO-format manifest file and save images to the specified directory.
        '''
        coco_dataset = {
            'info': {
                'description': 'ArTaxOr COCO-format dataset',
                'version': '1.0',
                'year': datetime.now().year,
                'date_created': datetime.now().isoformat()
            },
            'images': [],
            'annotations': [],
            'categories': [{'id': v, 'name': k} for k, v in self.cat_to_id.items()]
        }

        annotation_id = 1

        # Create the output directory for images if it doesn't exist
        images_output_dir = os.path.join(output_path, 'images')
        os.makedirs(images_output_dir, exist_ok=True)

        for pair_dict in self.generate_dataset_coco():
            # Save image to the output directory
            img_output_path = os.path.join(images_output_dir, pair_dict['file_name'])
            pair_dict['image'].save(img_output_path)

            # Add image metadata to the manifest
            coco_dataset['images'].append({
                'id': pair_dict['image_id'],
                'width': pair_dict['width'],
                'height': pair_dict['height'],
                'file_name': pair_dict['file_name'],
                'date_created': datetime.now().isoformat()
            })

            # Add annotations to the manifest
            for bbox, category_id, area in zip(pair_dict['objects']['bbox'], pair_dict['objects']['category'], pair_dict['objects']['area']):
                coco_dataset['annotations'].append({
                    'id': annotation_id,
                    'image_id': pair_dict['image_id'],
                    'category_id': category_id,
                    'bbox': bbox,
                    'area': area,
                    'iscrowd': 0
                })
                annotation_id += 1

        # Save the manifest file
        manifest_output_path = os.path.join(output_path, 'annotations.json')
        with open(manifest_output_path, 'w') as f:
            json.dump(coco_dataset, f, indent=4)

        print(f"COCO-format dataset created at {output_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate and save a COCO-format dataset.")
    parser.add_argument("--data_root_path", type=str, default='ArTaxOr', help="Path to the root directory of the dataset.")
    parser.add_argument("--output_path", type=str, default="./ArTaxOr_COCO_dataset", help="Path to save the generated dataset.")

    # Parse arguments
    args = parser.parse_args()

    # Set the current working directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Create an instance of the HF_Dataset_Generator class
    generator = HF_Dataset_Generator(args.data_root_path)

    # Create and save the COCO-format dataset
    generator.create_coco_manifest(args.output_path)