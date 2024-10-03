from transformers import AutoModelForObjectDetection, AutoImageProcessor, TrainingArguments, Trainer
from datasets import load_from_disk
import os

# Set the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the dataset from disk
dataset_path = "ArTaxOr_HF_dataset"
print(f"Loading dataset from {dataset_path}...")
dd = load_from_disk(dataset_path)

# Print the DatasetDict to verify
print(dd)

# Load the pre-trained model and image processor
hf_model_string = "microsoft/focalnet-base"
print(f"Loading model and image processor from {hf_model_string}...")
model = AutoModelForObjectDetection.from_pretrained(hf_model_string)
image_processor = AutoImageProcessor.from_pretrained(hf_model_string)

# Define the preprocessing function
def preprocess_function(examples):
    images = [example['image'] for example in examples]
    bboxes = [example['objects']['bbox'] for example in examples]
    labels = [example['objects']['category'] for example in examples]

    # Preprocess the images and annotations
    encoding = image_processor(images, annotations={"bbox": bboxes, "category_id": labels}, return_tensors="pt")

    return encoding

# Apply the preprocessing function to the dataset
print("Preprocessing the dataset...")
dd = dd.map(preprocess_function, batched=True, remove_columns=["image", "objects"])

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
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
    model=model,
    args=training_args,
    train_dataset=dd["train"],
    eval_dataset=dd["validation"],
    tokenizer=image_processor,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the model
print("Saving the model...")
model.save_pretrained("./focalnet-finetuned")
image_processor.save_pretrained("./focalnet-finetuned")

print("Training complete and model saved.")