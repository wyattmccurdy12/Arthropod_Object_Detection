# Arthropod object detection using the ArTaxOr dataset and FocalNet

## Goal
The goal of this repository is to fine-tune FocalNet to improve object detection on the [ArTaxOr](https://www.kaggle.com/datasets/mistag/arthropod-taxonomy-orders-object-detection-dataset/data) dataset. 

## Project Overview
This project focuses on improving classifications on the Arthropod Taxonomy Orders Object Detection Dataset (ArTaxOr) from Kaggle. ArTaxOr contains 15,376 images of arthropods from 7 different orders. Its images may contain multiple arthropods of different orders within one scene. All arthropod objects are labeled, and bounding box data is provided for each arthropod. 
In the only known publication on this dataset, YOLOX was used to achieve a mAP@50 of 90%, and a mAP@50:95 of %75.41. This is a good result, but still leaves room for improvement and exploration. The main limitation provided by the author was in situations where the object being detected was camoflaged, with the same color and texture as its immediate background. Breaking new ground on these edge cases promises to be challenging.


## Importance of the Problem
The phylum Arthropoda is the largest phylum in terms of both diversity and numbers in the animal kingdom. The phylum Arthropoda includes the subphylum Hexapoda, which contains the class Insecta, commonly known as insects. Insects have a large impact upon almost every aspect of human life, and can make or break quality of life in a surprising number of cases. Insects can affect human crops, human health, pets, housing, and many other important aspects of life. In addition to the impact that insects have upon human well-being, insects have a large amount of novelty, entertainment, and educational value for people.

## Approach
We propose to apply an advanced deep learning architecture, specifically FocalNet, to this problem and create a robust application for arthropod localization and identification. 
FocalNets, developed by Yang et. al. of Microsoft, are a type of neural network architecture for computer vision tasks. They offer an attention-free alternative to the popular self-attention mechanism found in Transformers. Instead of relying on attention, FocalNets employ a "focal modulation" process that prioritizes the aggregation of contextual information and then selectively interacts with it. This approach allows FocalNets to efficiently capture long-range dependencies in images while avoiding the computational overhead associated with self-attention. As a result, FocalNets have demonstrated impressive performance on various benchmarks, including image classification, object detection, and semantic segmentation, often surpassing self-attention-based models in terms of both accuracy and efficiency.
In the context of object detection on insects and other arthropods, FocalNet may be a promising way forward.

## Files
- `create_hf_dataset.py`: Script to generate and save the Hugging Face dataset -- and the coco formatted dataset as well.
- `train_model.py`: Script to train the FocalNet model on the ArTaxOr dataset.
- `README.md`: This file, providing an overview of the project.

## Usage
### Generate Dataset
To generate the dataset, run the following command:
```bash
python create_hf_dataset.py --data_root_path ArTaxOr --output_path ArTaxOr_HF_dataset
```

### Generate Model
To train the model, run the following command: 
```bash
python train_model.py --hf_model_name microsoft/focalnet-base  --dataset_path ArTaxOr_HF_dataset --output_dir focalnet-finetuned
```