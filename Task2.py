#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Importing Libraries and Loading Data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


# In[14]:


# Load the dataset
def load_data(file_path):
    return pd.read_csv("C:/Users/patha/OneDrive/Desktop/dataset/e commerce file.csv",encoding='ISO-8859-1')
data = load_data("C:/Users/patha/OneDrive/Desktop/dataset/e commerce file.csv")


# In[68]:


if data is not None:
    # Preprocessing function
    def preprocess_data(data):
        # Convert 'Quantity' and 'UnitPrice' to numeric
        data.loc[:, 'Quantity'] = pd.to_numeric(data['Quantity'], errors='coerce')
        data.loc[:, 'UnitPrice'] = pd.to_numeric(data['UnitPrice'], errors='coerce')
        return data
# Apply preprocessing function
processed_data = preprocess_data(data)

# Display the output
print(processed_data)


# In[67]:


# Data Preprocessing
def preprocess_data(data):
    data = data.dropna()
# Dropping rows with missing values
    return data

# Apply preprocessing
cleaned_data = preprocess_data(data)

# Display the output
print("\nData After Dropping Rows with Missing Values:")
print(cleaned_data)


# In[63]:


# Apply preprocessing
data = preprocess_data(data)
print("Data after preprocessing:")
print(data.head())


# In[66]:


# Feature Selection
def select_features(data):
    X = data[['UnitPrice']]  # Feature(s)
    y = data['Quantity']     # Target
    return X, y
X, y = select_features(data)
print("Features (X):")
print(X)
print("\nTarget (y):")
print(y)


# In[69]:


# Splitting the data into training and testing sets
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Apply the function
X_train, X_test, y_train, y_test = split_data(X, y)

# Display the results
print("X_train:")
print(X_train)
print("X_test:")
print(X_test)
print("y_train:")
print(y_train)
print("y_test:")
print(y_test)


# In[70]:


# Creating and training the linear regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)
# Display the model coefficients and intercept
print("Model Coefficients:")
print(model.coef_)
print("Model Intercept:")
print(model.intercept_)


# In[71]:


# Making predictions and evaluating the model's performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred

mse, r2, y_pred = evaluate_model(model, X_test, y_test)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")


# In[72]:


# Visualization
def visualize_results(X_test, y_test, y_pred):
    plt.figure(figsize=(10,6))
    plt.scatter(X_test, y_test, color='blue', label='Actual Values')
    plt.plot(X_test, y_pred, color='red', label='Regression Line')
    plt.xlabel('Unit Price')
    plt.ylabel('Quantity')
    plt.title('Regression Line: Actual vs Predicted')
    plt.legend()
    plt.show()
    
visualize_results(X_test, y_test, y_pred)


# In[79]:


def visualize_results(X_test, y_test, y_pred):
    plt.figure(figsize=(12, 8))

    # Scatter plot for actual values
    plt.scatter(X_test, y_test, color='green', label='Actual Values', alpha=0.7, edgecolors='k')

    # Regression line
    plt.plot(X_test, y_pred, color='yellow', linewidth=2, label='Regression Line')

    # Enhanced visualization features
    plt.xlabel('Unit Price', fontsize=14)
    plt.ylabel('Quantity', fontsize=14)
    plt.title('Regression Line: Actual vs Predicted', fontsize=16)
    plt.legend()
    
    # Adding grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)

    # Adding a regression line equation and R-squared value
    slope, intercept = np.polyfit(X_test.squeeze(), y_pred, 1)
    r_value = np.corrcoef(X_test.squeeze(), y_pred)[0, 1]
    plt.text(min(X_test.squeeze()), max(y_pred), f'Y = {slope:.2f}X + {intercept:.2f}\nR² = {r_value**2:.2f}', 
             fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

    # Show plot
    plt.show()

# Example usage:
# Assuming y_pred is the prediction from your model
# y_pred = model.predict(X_test)
visualize_results(X_test, y_test, y_pred)


# In[74]:


import joblib


# In[75]:


#Saving the Model
# Saving the trained model
def save_model(model, filename='linear_regression_model.pkl'):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")
save_model(model)


# # CONCLUSION
# 
# ### Conclusion in Points:
# 
# 1. Project Overview :
# Successfully implemented a simple linear regression model to predict the quantity of products sold based on unit price using an e-commerce dataset.
# 
# 2. Data Processing:
# Preprocessed the data by handling missing values and selecting relevant features for the model.
# 
# 3. Model Evaluation: 
# The model's performance was evaluated using Mean Squared Error (MSE) and R-squared (R²) metrics. The low R-squared value indicates that the linear model does not explain much of the variance in the target variable.
# 
# 4. Key Findings: 
# The relationship between unit price and quantity sold appears weak, suggesting that other factors may significantly impact sales quantities.
# 
# 5. Visualization: 
# Visualized the regression line against actual values, further confirming the model's limited predictive power.
# 
# 6. Future Work:
# To improve the model, consider incorporating additional features, performing feature engineering, or exploring more complex models.
# 
# 7. Learning Outcome: 
# This project highlights the importance of understanding the data and the limitations of simple models in capturing complex relationships.
