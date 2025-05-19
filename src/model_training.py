import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv('../data/simulated_housing_data.csv')

# Features and target
X = df.drop(columns='sale_price')
X = pd.get_dummies(X, columns=['zip_code'])
y = df['sale_price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print('MAE:', mean_absolute_error(y_test, y_pred))
print('R^2 Score:', r2_score(y_test, y_pred))
