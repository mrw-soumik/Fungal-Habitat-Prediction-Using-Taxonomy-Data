# Fungal Habitat Prediction

This project uses machine learning techniques to predict fungal habitats based on observational data, providing insights into ecological classification.

## Project Overview
The project includes data preprocessing, model training, evaluation, and visualizations to understand which features contribute most to predicting fungal habitats. The models used in this project include:
- **Random Forest Classifier**
- **LightGBM Classifier**

## Dataset
The dataset, `occurrences.csv`, includes fields such as:
- **family** and **genus**: Taxonomic information on fungal species.
- **month**: Month of observation.
- **habitat**: The type of habitat where fungi were found.
- **decimalLatitude** and **decimalLongitude**: Geographic location data.

### Files
- **notebook/Fungal_Habitat_Prediction.ipynb**: Main Jupyter notebook for analysis.
- **fungal_habitat_prediction.py**: Python file with the code for running the model outside the notebook.
- **data/occurrences.csv**: The dataset file used in this project.
- **images/**: Folder containing generated visualizations.

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/Fungal_Habitat_Prediction.git
   cd Fungal_Habitat_Prediction
