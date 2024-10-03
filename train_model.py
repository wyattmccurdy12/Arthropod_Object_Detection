'''
This script trains a FocalNet model on the ArTaxOr dataset. In the future, it should be able to train another model as well. 
The goal is to use command line arguments to create comparisons between object detection models.

sample command:
python train_model.py -d ArTaxOr_HF_dataset -h microsoft/focalnet-base -o focalnet_artaxor_output
'''

import os
import argparse
from transformers import TrainingArguments, Trainer
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from datasets import load_from_disk

class Arthropod_Focal_Net:
    def __init__(self, dataset_path, hf_model_string, output_dir):
        self.dataset_path = dataset_path
        self.hf_model_string = hf_model_string
        self.output_dir = output_dir
        self.dataset = None

    def load_dataset(self):
        print(f"Loading dataset from {self.dataset_path}...")
        try:
            self.dataset = load_from_disk(self.dataset_path)
        except:
            raise Exception("Dataset not found.")

    def load_model_and_processor(self):
        '''
        Load the model and image processor. The model is loaded from the Hugging Face model hub.
        '''
        self.model = AutoModelForObjectDetection.from_pretrained(self.hf_model_string)
        self.image_processor = AutoImageProcessor.from_pretrained(self.hf_model_string)

    def preprocess(self, examples):
        '''
        Get images, bounding boxes, and labels from the examples.
        Get encodings for images, bounding boxes, and labels.

        Args:
            examples: list of examples from the dataset
        
        Returns:
            encoding: dictionary containing encodings for images, bounding boxes, and labels
        '''
        images = [example['image'] for example in examples]
        bboxes = [example['objects']['bbox'] for example in examples]
        labels = [example['objects']['category'] for example in examples]

        # Preprocess the images and annotations
        encoding = self.image_processor(images, annotations={"bbox": bboxes, "category_id": labels}, return_tensors="pt")

        return encoding

    def preprocess_dataset(self):
        '''
        Apply the preprocess function to the dataset.
        '''
        print("Preprocessing the dataset...")
        self.dataset = self.dataset.map(self.preprocess, batched=True, remove_columns=["image", "objects"])

    def train_model(self):
        '''
        Train the model using a Hugging Face Trainer.
        '''
        # Define the training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=10,
            save_steps=10_000,
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=500,
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.image_processor,
        )

        # Train the model
        print("Starting training...")
        trainer.train()

        # Save the model
        print("Saving the model...")
        self.model.save_pretrained(self.output_dir)
        self.image_processor.save_pretrained(self.output_dir)
        print("Training complete and model saved.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train FocalNet model on ArTaxOr dataset.")
    parser.add_argument("--dataset_path", '-d', type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--hf_model_string", '-hf', type=str, required=True, help="Hugging Face model string.")
    parser.add_argument("--output_dir", '-o', type=str, required=True, help="Directory to save the trained model.")

    # Parse arguments
    args = parser.parse_args()

    # Set the current working directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Create an instance of the Arthropod_Focal_Net class
    arthropod_focal_net = Arthropod_Focal_Net(args.dataset_path, args.hf_model_string, args.output_dir)

    # Load dataset
    arthropod_focal_net.load_dataset()

    # Load model and processor
    arthropod_focal_net.load_model_and_processor()

    # Preprocess dataset
    arthropod_focal_net.preprocess_dataset()

    # Train model
    arthropod_focal_net.train_model()

if __name__ == "__main__":
    main()