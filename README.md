# Leveraging Contrastive Learning with SimSiam for the Classification of Primary and Secondary Liver Cancers

## Abstract
Accurate liver cancer classification is vital for effective treatment and patient prognosis. This project utilizes SimSiam, a self-supervised learning approach, to improve classification accuracy of liver tumors. We integrate SimSiam with three CNN classifiers - Inception, Xception, and ResNet152 - and pretrain them using two loss functions: MSE and COS. Our tests show consistent improvements in classification accuracy across all models. The dataset includes CT scans of 460 patients with various types of liver cancers.


<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1WYO3yMzypwe-1d94L6eLE7AXhszuf4Xz" alt="Your Image Description">
</p>


## Quick Start
- Clone the repository: git clone https://github.com/your_username/liver-cancer-classification.git
- Navigate to the project directory: cd liver-cancer-classification
- Install dependencies: pip install -r requirements.txt
- Train and evaluate the model: python train.py
- Evaluate the model: python evaluate.py
  

## Dependencies
- Python 3.8+
- TensorFlow 2.x
- NumPy
- scikit-learn

## Contribution
Feel free to open an issue or submit a PR for improvements. To access the datasets, please contact at Amber.Simpson@Queensu.ca

