# Fungal Habitat Prediction Using Taxonomy and Environmental Data

## Objective
This project classifies fungal habitats based on taxonomy (family, genus), geographic coordinates, and environmental conditions, using machine learning techniques. The model helps identify potential habitats for various fungi species, aiding ecological research and conservation.

## Dataset
The dataset, `occurrences.csv`, includes:
- Family, Genus
- Habitat type, Substrate type
- Geographic coordinates (latitude and longitude)
- Month of observation

## Project Structure
- **data/**: Contains the dataset (`occurrences.csv`).
- **notebooks/**: Jupyter notebooks for data preprocessing and visualization.
- **src/**: Python scripts for data processing and model training.
- **results/**: Visualizations of genus and family distributions.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/Fungal_Habitat_Prediction.git
cd Fungal_Habitat_Prediction
pip install -r requirements.txt
