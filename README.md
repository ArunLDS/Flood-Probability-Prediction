# Flood Probability Prediction (Regression Model)

## Overview
The goal of this project is to build a machine learning model that can predict the probability of flood occurrence. Several models were trained for this task, and the final predictions were made using the `Stacking method`. This notebook was also used in a Kaggle competition.   
[View this notebook in kaggle](https://www.kaggle.com/code/arunl15/ensemble-model-xgb-lgbm-cbr-xgbrf-ridge)

## Dataset Info
The dataset for this competition (both train and test) was generated from a deep learning model trained on a separate Flood `Prediction Factors dataset`. Feature distributions are close to, but not exactly the same as the original. (info from from Kaggle)     
[Click here to view the dataset](https://www.kaggle.com/competitions/playground-series-s4e5/data)

## EDA
- The dataset consists of 1,117,957 entries and 20 independent features.
- No null values or duplicates are found in the data.
- Mean and Std of all columns appear to be very similar.
- There are no continuous values present at all in the dataset. All features contain numbers between the range of 0 and 19. It is very much possible that all of these are categorical in nature.
- The correlation matrix shows that all the features are slightly negatively correlated with each other and all of them are positively correlated with the dependent Target variable.
- PCA was applied on the dataset and it was found that each feature captures roughly the same amount of variance. Since PCA is unable to transform the data by capturing the greatest variance of the data in the first few principal components, it is not useful in this scenario.

## Model Training
- The initial baseline performace for the models were found using cross validation.
- Initial analysis revealed that feature sums between 72 and 75 show higher average FloodProbability. A boolean column (linear_ft) was created to indicate this range which will be helpful for training Linear machine learning models.
- A sum column that contains the sum of the features along the column axis was also created, which will be helpful for tree-based models. Several other feature were also engineered.
- Following feature engineering, there was a notable improvement in model performance.
- Subsequently, several machine learnin models including XGBoost, CatBoost, LightGBM, RandomForest model of XGBoost and Ridge were trained and and hyperparameters were fine-tuned using Optuna.
- Continuing further, the predictions for both train and test data was made using cross validation and different random_states so as to ensure robust predictions without overfitting.
- The predictions were stacked and Optuna was utilized again to optimize the blending (weight combinations) of these predictions for achieving final predictions.
