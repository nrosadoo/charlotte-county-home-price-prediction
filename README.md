# Charlotte County Home Price Prediction

This project builds a machine learning model to predict home sale prices in Charlotte County, FL using simulated housing data.

##  Dataset
The dataset includes:
- Bedrooms, Bathrooms
- Square Footage
- Lot Size
- Year Built
- Zip Code
- Days on Market
- Sale Price

##  Models
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

##  Technologies
Python, pandas, scikit-learn, matplotlib, seaborn, xgboost

##  How to Use
1. Explore the data in `notebooks/data_exploration.ipynb`
2. Train models using `src/model_training.py`
3. Install dependencies: `pip install -r requirements.txt`

##  Structure
```
charlotte-county-home-price-prediction/
├── data/
│   └── simulated_housing_data.csv
├── notebooks/
│   └── data_exploration.ipynb
├── src/
│   └── model_training.py
├── README.md
└── requirements.txt
```
