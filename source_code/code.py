# ==============================
# HOUSE PRICE PREDICTION MODEL
# Linear Regression (ML Project)
# ==============================

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import pickle


# ==============================
# 1. CREATE DATASET
# ==============================
data = {
    "sqft": [800, 900, 1000, 1100, 1200,
             1300, 1400, 1500, 1600, 1700,
             1800, 1900, 2000, 2100, 2200,
             2300, 2400, 2500, 2600, 2700,
             2800, 2900, 3000, 3200, 3400,
             3600, 3800, 4000, 4200, 4500],

    "bedrooms": [1, 2, 2, 2, 3,
                 3, 3, 3, 3, 3,
                 4, 4, 4, 4, 4,
                 4, 5, 5, 5, 5,
                 5, 5, 6, 6, 6,
                 6, 7, 7, 7, 8],

    "bathrooms": [1, 1, 1, 2, 2,
                  2, 2, 2, 2, 3,
                  3, 3, 3, 3, 3,
                  4, 4, 4, 4, 4,
                  4, 5, 5, 5, 5,
                  6, 6, 6, 7, 7],

    "price": [120000, 150000, 180000, 200000, 230000,
              250000, 270000, 300000, 320000, 350000,
              380000, 400000, 430000, 450000, 480000,
              510000, 540000, 570000, 600000, 630000,
              660000, 690000, 720000, 780000, 830000,
              880000, 930000, 980000, 1030000, 1100000]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())


# ==============================
# 2. SPLIT FEATURES & TARGET
# ==============================
X = df[['sqft', 'bedrooms', 'bathrooms']]
y = df['price']


# ==============================
# 3. TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 4. MODEL TRAINING
# ==============================
model = LinearRegression()
model.fit(X_train, y_train)


# ==============================
# 5. PREDICTION
# ==============================
y_pred = model.predict(X_test)

print("\nPredicted Prices:")
print(y_pred)

print("\nActual Prices:")
print(y_test.values)


# ==============================
# 6. MODEL EVALUATION
# ==============================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# ==============================
# 7. PREDICT NEW HOUSE
# ==============================
new_house = pd.DataFrame({
    "sqft": [2500],
    "bedrooms": [5],
    "bathrooms": [4]
})

predicted_price = model.predict(new_house)

print("\nPredicted Price for New House:")
print(predicted_price[0])


# ==============================
# 8. VISUALIZATION
# ==============================
plt.scatter(df['sqft'], df['price'])
plt.xlabel("Square Footage")
plt.ylabel("House Price")
plt.title("House Price Prediction")
plt.show()


# ==============================
# 9. SAVE MODEL
# ==============================
pickle.dump(model, open("house_price_model.pkl", "wb"))


# ==============================
# 10. LOAD MODEL & TEST
# ==============================
loaded_model = pickle.load(open("house_price_model.pkl", "rb"))

prediction = loaded_model.predict(new_house)

print("\nPrediction using Saved Model:")
print(prediction[0])