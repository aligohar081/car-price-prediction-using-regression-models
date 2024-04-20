# car-price-prediction-using-regression-models
Building a Machine Learning Model to Predict the Price of the Car By Comparing Performance of Different Regression Techniques (Simple Linear Regression, Multiple Linear Regression, Polynomial Regression)
<br>
<i>Comparing these three models, we conclude that the <b>Multiple Linear Regression</b> model is the best model to be able to predict price from our dataset.</i> This result makes sense, since we have 27 variables in total, and we know that more than one of those variables are potential predictors of the final car price.

These are the changes i have done for improving privacy

1. **Data Preprocessing**:
   - The new code includes a check for missing values using `df.isnull().sum()`.
   - Rows with missing values are dropped using `df.dropna(inplace=True)`. This ensures that the dataset used for training the models does not contain any missing values.

2. **Model Training and Evaluation**:
   - The data is split into training and testing sets using `train_test_split` from `sklearn.model_selection`.
   - Both linear regression (`LinearRegression`) and polynomial regression (`PolynomialFeatures` with `LinearRegression`) models are trained on the training data and evaluated on the testing data.
   - Mean squared error (`mean_squared_error`) and R-squared (`r2_score`) are used as evaluation metrics for both models. These metrics are calculated for both linear and polynomial regression models.

3. **Visualization**:
   - The scatter plot of the testing data along with the regression lines of both linear and polynomial regression models is visualized using `matplotlib`.

These changes aim to improve the accuracy and robustness of the analysis by incorporating data preprocessing, proper evaluation on a separate testing set, and visualization of the model predictions.
