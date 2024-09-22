# Assignment 1 - SBU Machine Learning Course (Winter 2023)

## Description

This repository contains the solutions and implementation for the first assignment of the **Machine Learning Course** at Shahid Beheshti University (Winter 2023). The assignment covers a range of machine learning concepts including regression, regularization techniques, gradient descent, model evaluation, and feature selection. 

The main focus of the tasks is both theoretical analysis and practical implementation. The implementations utilize Python and the Scikit-learn library, and are supplemented with custom algorithms developed from scratch.

---

## Assignment Tasks Overview

1. **Gradient Descent in Logistic Regression**  
   * Can gradient descent get stuck in local minima in logistic regression? Explore and explain why.

2. **Polynomial Regression and Learning Curves**  
   * Analyze the gap between training error and validation error in polynomial regression and suggest methods to resolve it.

3. **Ridge Regression: High Bias or High Variance?**  
   * Assess model performance and adjust the regularization hyperparameter.

4. **Comparison of Regression Techniques**  
   * Discuss when to use ridge, lasso, and elastic net regression.

5. **Linear Regression with Mean Absolute Error**  
   * Implement linear regression using MAE as the cost function from scratch and compare results with Scikit-learn.

6. **Linear Regression using Normal Equation**  
   * Implement linear regression using the normal equation from scratch.
   
7. **Bootstrapping vs Cross-Validation**  
   * Compare bootstrapping and cross-validation. Discuss when each method is appropriate.

8. **Nested Cross-Validation & Statistical Significance (Extra Points)**  
   * Provide detailed explanations of nested cross-validation, 5x2 cross-validation, and statistical significance tests.

9. **Forward and Backward Feature Selection**  
   * Implement forward and backward feature selection algorithms from scratch, using MSE as the evaluation metric.

10. **Scaling and Model Sensitivity**  
    * Investigate the sensitivity of gradient descent, normal equations, and SVD algorithms to feature scaling.

11. **News Popularity Prediction Dataset**  
    * Build a regression model to predict article popularity using Scikit-learn. Perform EDA, apply ridge and lasso regression, and explore scaling, polynomial features, and hyperparameter tuning techniques.
    
12. **Batch Gradient Descent with Early Stopping for Softmax Regression (Extra Points)**  
    * Implement batch gradient descent with early stopping for softmax regression, and apply it to the Penguins dataset.

---

## Files in This Repository

| File Name                                   | Description                                                             |
|---------------------------------------------|-------------------------------------------------------------------------|
| `Question 12 (2).ipynb`                     | Implementation of question 12 - News Popularity Prediction.             |
| `Question 13 (1).ipynb`                     | Implementation of question 13 - Batch gradient descent with early stopping for softmax regression. |
| `question 10 (1).ipynb`                     | Implementation of question 10 - Forward and Backward Feature Selection. |
| `question 6 (1).ipynb`                      | Implementation of question 6 - Linear Regression using Normal Equation. |
| `assignment 1 - code reports.pdf`           | Persian report detailing the implementation of the code.                |
| `assignment 1 - code reports in English.pdf`| English report detailing the implementation of the code.                |

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/SBU_Machine_Learning_Assignment_1.git
   ```

2. Navigate to the project directory:
   ```bash
   cd SBU_Machine_Learning_Assignment_1
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Explore the individual notebooks to review the implementation of each task:
   - Question 6: Linear Regression using the Normal Equation
   - Question 10: Feature Selection
   - Question 12: News Popularity Prediction
   - Question 13: Softmax Regression with Batch Gradient Descent

5. Read the detailed code reports for insights into the approaches and results for each question.
