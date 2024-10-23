'''
This script trains a Mask R-CNN model with FocalNet as the backbone on the ArTaxOr dataset. 
The goal is to use command line arguments to create comparisons between object detection models.

sample command:
python train_model.py -d ArTaxOr_COCO -hf microsoft/focalnet-base -o focalnet_artaxor_output
'''

# Basic imports
import os
import argparse
import numpy as np

# Torch imports
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Huggingface imports
from transformers import AutoImageProcessor, AutoModel

# Define the transform pipeline
transform = A.Compose([
    A.Resize(480, 480),
    A.HorizontalFlip(p=1.0),
    A.RandomBrightnessContrast(p=1.0),
    ToTensorV2()
], bbox_params=A.BboxParams(format='coco', label_fields=['category_id']))

class CocoDataset(torchvision.datasets.CocoDetection):
    """
    Custom dataset class for loading COCO dataset with additional transformations.

    Args:
        root (str): Root directory where images are downloaded to.
        annFile (str): Path to the annotation file.
        transforms (callable, optional): A function/transform that takes in an image and its bounding boxes and returns a transformed version.
    """
    def __init__(self, root, annFile, transforms=None):
        super(CocoDataset, self).__init__(root, annFile)
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: (image, target) where target is a list of dictionaries containing bounding boxes and category IDs.
        """
        # Get item from the COCO dataset
        img, target = super(CocoDataset, self).__getitem__(idx)

        # Numpy array of image
        img = np.array(img.convert("RGB"))

        # Get bboxes
        bboxes = [obj['bbox'] for obj in target]

        # Get category IDs
        category_ids = [obj['category_id'] for obj in target]

        # Apply transforms if indeed they have been defined and provided
        if self.transforms:
            # Apply the transformations with named arguments
            transformed = self.transforms(image=img, bboxes=bboxes, category_id=category_ids)
            img = transformed['image']
            target = [{'bbox': bbox, 'category_id': category_id} for bbox, category_id in zip(transformed['bboxes'], transformed['category_id'])]
        return img, target

class Arthropod_Focal_Net:
    """
    Class for training a Mask R-CNN model with a FocalNet backbone on the ArTaxOr dataset.

    Args:
        dataset_path (str): Path to the dataset.
        hf_model_string (str): Hugging Face model string for the FocalNet backbone.
        output_dir (str): Directory to save the trained model.
        sample_percent (float, optional): Percentage of the dataset to sample for debugging.
    """
    def __init__(self, dataset_path, hf_model_string, output_dir, sample_percent=None):
        """
        Initialize the Arthropod_Focal_Net class with dataset path, model string, output directory, and sample percentage.

        Args:
            dataset_path (str): Path to the dataset.
            hf_model_string (str): Hugging Face model string for the FocalNet backbone.
            output_dir (str): Directory to save the trained model.
            sample_percent (float, optional): Percentage of the dataset to sample for debugging.
        """
        self.dataset_path = dataset_path
        self.hf_model_string = hf_model_string
        self.output_dir = output_dir
        self.sample_percent = sample_percent
        self.dataset = None

    def load_dataset(self):
        """
        Load the dataset from the specified path and apply transformations. Optionally sample a percentage of the dataset for debugging.
        """
        print(f"Loading dataset from {self.dataset_path}...")
        annotationFile = os.path.join(self.dataset_path, 'annotations.json')
        root_path = os.path.join(self.dataset_path, 'images')
        # self.dataset = CocoDataset(root=root_path, annFile=annotationFile, transforms=transform)
        self.dataset = CocoDataset(root = root_path, annFile = annotationFile)

        if self.sample_percent:
            print(f"Sampling {self.sample_percent}% of the dataset for debugging...")
            num_samples = int(len(self.dataset) * (self.sample_percent / 100))
            self.dataset = Subset(self.dataset, range(num_samples))

    def load_model_and_processor(self):
        """
        Load the FocalNet model and image processor from the Hugging Face model hub. 
        Create the Mask R-CNN model using FocalNet as the backbone.

        The FocalNet model is using transformers methods, while the Mask R-CNN model is using torchvision methods.

        """
        # Load the backbone model and its image processor
        self.backbone = AutoModel.from_pretrained(self.hf_model_string)
        self.image_processor = AutoImageProcessor.from_pretrained(self.hf_model_string)

        # Create the Mask R-CNN model using FocalNet as the backbone
        backbone = self.backbone
        backbone.out_channels = 2048 

        # Create an anchor generator for the FPN
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        # Create a RoI align layer
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2
        )

        # Create the Mask R-CNN model
        self.model = MaskRCNN(
            backbone,
            num_classes=8,  # 7 classes + background
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

    def collate_fn(self, batch):
        """
        Custom collate function to handle batches of images and targets.

        Args:
            batch (list): List of tuples containing images and targets.

        Returns:
            tuple: Tuple of lists containing images and targets.
        """
        return tuple(zip(*batch))

    def train_model(self):
        """
        Train the Mask R-CNN model using PyTorch. The model is trained for a specified number of epochs and the training and validation losses are printed.

        The trained model is saved to the specified output directory.
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)

        # Create data loaders
        train_loader = DataLoader(self.dataset, batch_size=4, shuffle=True, collate_fn=self.collate_fn)

        # Define the optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            self.model.train()
            for images, targets in train_loader:


                '''
                In comes a list for images -- a list for targts

                The images list contains numpy arrays
                The targets list contains a list of dictionaries,
                unfortunately the dictionaries are in lists of length 1 for some reason... TODO: Fix this
                '''

                images = list(torch.tensor(image).to(device) for image in images)
                
                # Fix targets to be a list of dictionaries
                targets = [target[0] if isinstance(target, list) and len(target) == 1 else target for target in targets]


                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")

            # Validation loop
            self.model.eval()
            with torch.no_grad():
                for images, targets in train_loader:
                    images = list(image.to(device) for image in images)
                    targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    print(f"Validation Loss: {losses.item()}")

        # Save the model
        print("Saving the model...")
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "model.pth"))
        print("Training complete and model saved.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train Mask R-CNN model with FocalNet backbone on ArTaxOr dataset.")
    parser.add_argument("--dataset_path", '-d', type=str, default='ArTaxOr_COCO', help="Path to the COCO dataset.")
    parser.add_argument("--hf_model_string", '-hf', type=str, default='microsoft/focalnet-base', help="Hugging Face model string.")
    parser.add_argument("--output_dir", '-o', type=str, default='focalnet_maskrcnn', help="Directory to save the trained model.")
    parser.add_argument("--sample_percent", '-sp', type=int, default=10, help="Percentage of samples to preprocess for debugging (1-100).")

    # Parse arguments
    args = parser.parse_args()

    # Set the current working directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Create an instance of the Arthropod_Focal_Net class
    arthropod_focal_net = Arthropod_Focal_Net(args.dataset_path, args.hf_model_string, args.output_dir, args.sample_percent)

    # Load dataset
    arthropod_focal_net.load_dataset()

    # Load model and processor
    arthropod_focal_net.load_model_and_processor()

    # Train model
    arthropod_focal_net.train_model()

if __name__ == "__main__":
    main()