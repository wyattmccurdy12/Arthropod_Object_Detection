import os
import json
from glob import glob
from PIL import Image
from datasets import Dataset, DatasetDict
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
            'objects': objects
        }

        return pair_dict

    def generate_dataset(self):
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

    def create_and_save_dataset(self, output_path):
        # Create the dataset from the generator
        print("Creating the dataset from the generator...")
        ds = Dataset.from_generator(self.generate_dataset)

        # Define the split ratios
        print("Splitting the dataset into train, test, and validation sets...")
        train_test_split = ds.train_test_split(test_size=0.2)
        test_valid_split = train_test_split['test'].train_test_split(test_size=0.5)

        # Create a DatasetDict
        dd = DatasetDict({
            'train': train_test_split['train'],
            'test': test_valid_split['test'],
            'validation': test_valid_split['train']
        })

        # Save the dataset to disk
        print(f"Saving the dataset to {output_path}...")
        dd.save_to_disk(output_path)
        print("Dataset saved successfully.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate and save a Hugging Face dataset.")
    parser.add_argument("--data_root_path", type=str, required=True, help="Path to the root directory of the dataset.")
    parser.add_argument("--output_path", type=str, default="./ArTaxOr_HF_dataset", help="Path to save the generated dataset.")

    # Parse arguments
    args = parser.parse_args()

    # Set the current working directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Create an instance of the HF_Dataset_Generator class
    generator = HF_Dataset_Generator(args.data_root_path)

    # Create and save the dataset
    generator.create_and_save_dataset(args.output_path)