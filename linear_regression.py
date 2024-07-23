import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create a dataset
data = {
    'square_footage': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
    'bedrooms': [3, 3, 3, 4, 4, 4, 5, 5, 5, 5],
    'bathrooms': [2, 2, 2, 3, 3, 3, 4, 4, 4, 4],
    'price': [300000, 320000, 340000, 400000, 420000, 440000, 500000, 520000, 540000, 560000]
}
df = pd.DataFrame(data)

# Display the dataset
print("Dataset:")
print(df)

# Separate features and target
X = df[['square_footage', 'bedrooms', 'bathrooms']]
y = df['price']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nTraining Features (X_train):")
print(X_train)
print("\nTesting Features (X_test):")
print(X_test)
print("\nTraining Target (y_train):")
print(y_train)
print("\nTesting Target (y_test):")
print(y_test)
print("\nMean Squared Error:", mse)
print("R^2 Score:", r2)

# Plot the results
plt.figure(figsize=(10, 6))

# Plot training data
plt.scatter(X_train['square_footage'], y_train, color='blue', label='Training data')
# Plot testing data
plt.scatter(X_test['square_footage'], y_test, color='green', label='Testing data')
# Plot predicted data
plt.scatter(X_test['square_footage'], y_pred, color='red', label='Predicted prices')

plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('House Price Prediction using Linear Regression')
plt.legend()
plt.show()

# Example: Predict the price of a house with 2000 square feet, 4 bedrooms, and 3 bathrooms
example_house = pd.DataFrame([[2000, 4, 3]], columns=['square_footage', 'bedrooms', 'bathrooms'])
predicted_price = model.predict(example_house)
print(f"\nPredicted price for the example house: ${predicted_price[0]:,.2f}")
