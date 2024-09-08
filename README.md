# Name : Garima Pathak
# Company :CODTECH IT Solutions
# ID: CT08DS7035
# Domain: Data Analytics
# Duration: AUGUST 17th, 2024 to SEPTEMBER 17th, 2024.

### E-Commerce Sales Prediction using Linear Regression

## **Project Objective**
The goal of this project is to develop a simple linear regression model to predict the quantity of products sold based on their unit price. The dataset used comes from an e-commerce platform, and the project aims to understand the relationship between pricing and sales.

## **Key Activities**
1. **Data Preprocessing**: 
   - Handled missing values.
   - Converted relevant features like "Quantity" and "UnitPrice" to numeric formats for proper model training.
   - Dropped rows with missing values to clean the dataset.

2. **Feature Selection**: 
   - Selected "UnitPrice" as the feature (X) and "Quantity" as the target variable (y) for the model.

3. **Model Training & Evaluation**: 
   - Split the data into training and testing sets.
   - Trained a linear regression model.
   - Evaluated the model using Mean Squared Error (MSE) and R-squared (R²) metrics.

4. **Visualization**: 
   - Visualized the linear regression line against the actual data to assess the model’s performance.

5. **Model Saving**: 
   - Saved the trained model using Joblib for future use

## **Technology Used**
- **Languages**: Python
- **Libraries**:
  - `Pandas` for data handling and preprocessing.
  - `Scikit-learn` for model training, evaluation, and splitting data.
  - `Matplotlib` for visualization.
  - `Joblib` for saving the model.
  

## **Key Insights**
- **Weak Relationship**: The model displayed a low R-squared value, indicating that the linear relationship between unit price and quantity sold is not strong. This suggests other factors beyond price may have a more significant impact on sales quantities.
  
- **Room for Improvement**: The model’s limited predictive power indicates the need to incorporate additional features (e.g., customer demographics, seasonality) and explore more complex models to improve accuracy.


## **Conclusion**
This project provided valuable insights into the process of building a linear regression model for sales prediction. While the current model shows limitations, it highlights the importance of feature selection and model evaluation in understanding the data. Future work includes expanding the feature set and exploring more advanced machine learning algorithms to capture the complexity of sales behavior more effectively.


### **Future Work**
1. Incorporate additional features to improve predictive accuracy.
2. Explore more advanced algorithms like decision trees or random forests to model the non-linear relationships in the data.
