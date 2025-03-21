# King County Housing Prices Prediction

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Running the Project](#running-the-project)
- [Dataset Description](#dataset-description)
- [Key Steps](#key-steps)
- [Results](#results)
- [Usage](#usage)

---

## Project Overview

This project aims to analyze and predict housing prices in King County, USA. As a Data Analyst working at a Real Estate Investment Trust, the objective is to determine the market price of a house based on various features such as square footage, number of bedrooms, number of floors, and other relevant attributes. The project leverages data analysis and machine learning techniques to build an accurate predictive model.

## Technologies Used

- **Python**: Programming language used for data analysis and modeling.
- **Libraries**:
  - `pandas`: Data manipulation and analysis.
  - `numpy`: Numerical computations.
  - `matplotlib` and `seaborn`: Data visualization.
  - `scikit-learn`: Machine learning algorithms and preprocessing tools.

## Running the Project

1. Clone this repository.
2. Open `House_Sales_in_King_Count_USA.ipynb` in Jupyter Notebook or JupyterLab.
3. Ensure the following libraries are installed: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.

## Dataset Description

The dataset contains house sale prices for King County, USA, including Seattle. It consists of various features such as:

- **sqft_living**: Square footage of the living space.
- **bedrooms**: Number of bedrooms.
- **bathrooms**: Number of bathrooms.
- **floors**: Number of floors in the house.
- **waterfront**: Whether the house is facing a waterfront (1 = yes, 0 = no).
- **condition**: Condition rating of the house.
- **grade**: Construction quality and design grade.
- **price**: Sale price of the house (target variable).

> **Note**: The dataset is publicly available and included in the project files.

## Key Steps

1. **Exploratory Data Analysis (EDA):**
   - Examined data distributions, missing values, and outliers.
   - Visualized correlations between features using heatmaps and scatter plots.

2. **Data Preprocessing:**
   - Handled missing values and outliers.
   - Normalized features using StandardScaler.
   - Split data into training and testing sets (80/20 split).

3. **Model Building:**
   - Developed a Linear Regression model using `scikit-learn`.
   - Trained the model on the training set and evaluated its performance on the test set.

4. **Evaluation Metrics:**
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - R-squared (R²)

## Results

- The model demonstrated a strong correlation between predicted and actual prices.

Visualizations include:
- Feature importance plots.
- Heatmaps showing correlations between key variables.
- Predicted vs. actual price comparisons.

## Usage

1. Run the Jupyter Notebook:
   ```bash
   jupyter notebook House_Sales_in_King_Count_USA.ipynb
   ```

2. Follow the steps outlined in the notebook to explore the dataset, preprocess the data, and train the model.

3. Modify or extend the notebook to experiment with different algorithms or hyperparameters.
