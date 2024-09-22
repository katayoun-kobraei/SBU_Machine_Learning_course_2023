# Machine Learning - 2nd Assignment

## Overview
This repository contains the materials for the second assignment of the Machine Learning course. The assignment includes theoretical questions regarding SVM classifiers, decision trees, ensemble methods, and practical implementation on the Vehicle Insurance Claim Fraud Detection dataset.

## Assignment Contents

### Theoretical Questions
1. **SVM Classifier Confidence Scores**: Discussion on whether SVM classifiers can provide confidence scores or probabilities.
2. **SVM Hyperparameter Tuning**: Recommendations for adjusting parameters when an SVM with an RBF kernel underfits.
3. **ϵ-Insensitive Models**: Explanation of what it means for a model to be ϵ-insensitive.
4. **Margin Types in SVM**: Differences between hard margin and soft margin SVM and appropriate use cases for each.
5. **Gini Impurity**: Analysis of Gini impurity in nodes compared to their parents.
6. **Scaling Features in Decision Trees**: Discussion on the necessity of scaling input features if a Decision Tree underfits.
7. **Feature Selection with Tree-Based Models**: Methods for using tree-based models for feature selection.
8. **Hyperparameter Tweaking**: Strategies for adjusting hyperparameters in AdaBoost and Gradient Boosting under specific conditions.
9. **Ensemble Types**: Differences between homogeneous and heterogeneous ensembles, and which is generally more powerful.
10. **ROC and AUC**: Their roles in evaluating classification performance.
11. **Threshold Values in Classification**: The impact of threshold values on model performance, including precision and recall trade-offs.
12. **Multiclass Classification Approaches**: Differences between one-vs-one and one-vs-all methods, and when to use each.

### Practical Implementation
13. **Vehicle Insurance Claim Fraud Detection**:
    - **Exploratory Data Analysis**: Initial analysis of the dataset.
    - **Models Implemented**: 
        - Logistic Regression
        - SVM
        - Decision Trees
        - Random Forest
        - Other classifiers (KNN, Naive Bayes, Ensemble models)
    - **Techniques**: Use of stratified cross-validation, handling class imbalance, and boosting performance through hyperparameter tuning and feature engineering.

14. **SVM for Anomaly Detection**: Discussion of using SVM for anomaly detection and associated challenges (Extra Point).
15. **Bagging Classifier Implementation**: Implementation of a Bagging classifier from scratch and testing it on the Penguins dataset (Extra Point).
16. **Class Imbalance in Ensemble Learning**: Techniques for handling class imbalance in ensemble methods, with explanations (Extra Point).

## Repository Files
- **`Assignment 2 (2).pdf`**: Document containing the assignment questions and requirements.
- **`Implementation Report q13 in English.pdf`**: Detailed implementation report for question 13 in English.
- **`q13 (2) (2).ipynb`**: Jupyter notebook containing the code for question 13.
- **`q15 (1).ipynb`**: Jupyter notebook for implementing the Bagging classifier (question 15).
- **`report-q13.pdf`**: Implementation report for question 13 in Persian.

## Getting Started

### Prerequisites
- Python 3.x
- Jupyter Notebook
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/[katayoun-kobraei]/Machine-Learning-Assignment-2.git
   cd Machine-Learning-Assignment-2
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebooks for implementation:
   ```bash
   jupyter notebook q13 (2) (2).ipynb
   jupyter notebook q15 (1).ipynb
   ```

## Conclusion
This assignment provided a comprehensive exploration of machine learning techniques and their application to real-world problems. Through both theoretical and practical tasks, students developed a deeper understanding of model evaluation, tuning, and deployment.

## Acknowledgments
Special thanks to the instructors and peers for their support throughout this assignment.
```
