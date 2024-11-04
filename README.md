# Fungal Habitat Prediction

This project predicts fungal habitats using a dataset of observed fungal occurrences. It employs machine learning models and various encoding techniques to classify habitat types effectively.

## Project Overview
Using Python and machine learning libraries, this project performs:
1. Data Cleaning and Preprocessing
2. Model Training (Random Forest, LightGBM)
3. Evaluation and Visualization of Model Performance

## Dataset
The dataset, `occurrences.csv`, contains:
- **family**, **genus**: Taxonomic data on fungi
- **month**: Month of observation
- **habitat**: Habitat type
- **decimalLatitude**, **decimalLongitude**: Location data

### Files
- **notebook/Fungal_Habitat_Prediction.ipynb**: Jupyter notebook containing all code.
- **data/occurrences.csv**: Data file used for training the model.
- **images/**: Folder to save and view generated plots.
- **fungal_habitat_prediction.py**: Python script for model training and evaluation.

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/Fungal_Habitat_Prediction.git
   cd Fungal_Habitat_Prediction
