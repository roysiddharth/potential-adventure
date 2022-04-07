# Titanic: Machine Learning From Disaster (Kaggle Competition)

Build and Train machine learning models to predict whether passengers survived or not.

## Author

[Siddhartha Roy](https://github.com/roysiddharth)

## Acknowledgements

- [Titanic - Machine Learning from Disaster (Kaggle Competition)](https://www.kaggle.com/c/titanic)
- [A beginner’s guide to Kaggle’s Titanic problem](https://towardsdatascience.com/a-beginners-guide-to-kaggle-s-titanic-problem-3193cb56f6ca)

## Project Pipeline

- Problem Statment
- Analytical Approach
- Data Requirements
- Data Collection
- Data Understanding
- Data Preparation
- Modeling
- Evaluation and Interpretation

## Libraries Used

- **Numpy** - support for large, multi-dimensional arrays and matrices.
- **Pandas** - for data manipulation and analysis
- **Seaborn** - statistical data visualization
- **Matplotlib** - library for creating static, animated, and interactive visualizations
- **Scikit-learn** - tools for predictive data analysis and machine learning

`pip install numpy, pandas, seaborn, matplotlib, sklearn`

## Business Understanding


The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” _RMS Titanic_ sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this project, I have tried to train a machine to predict whether the passengers survived based on these several factors and features of their travel.

## Analytical Approach

Our main aim is to predict the categorical variable (survived/not survived) with maximum accuracy. Hence we need a binary classification model for this task.

## Data Requirements

We require onboard passengers information which might include name, age, fare, gender, class, number of people traveling with, deck of embarkation etc.

## Data Collection

We are given both the datasets (training and testing) which are .csv files.
The train.csv file is used to train the model.
The test.csv file is used to test if our model can determine survival based on observations, not having the survival info.

## Data Understanding

This step is part of Exploratory Data Analysis.

There are 891 observations in the training dataset with each having 12 columns. 11 of them are predictor variables and 1 being target variable.

There are few different types of variables available.

- Continous: Age, Fare
- Discrete: SibSp, Parch
- Categorical: Survived, Sex, and Embarked
- Ordinal: Pclass
- Mixed: Ticket
- Alphanumeric: Cabin

There were 3 features having missing values.

- Cabin
- Age
- Embarked

Cabin has way too many missing values. We have imputed the missing values with the most frequent value and then extracted the deck information from that.

As per the training dataset, there were more male present compared to female and most of the people didnot survive. But females had better survival rate than males. It was also found out that survived passengers had paid more fare than the ones that did not survive. At the same time Pclass = 1 had better survival rate than the rest 2 classes.

Majority of the passengers were between the age group 15-35 but most of them didnot survived. Children aged < 4 and old aged people had higher survival rate.

## Data Preparation/Feature Engineering

After closely looking into the dataset, variables types, values, amount of missing values present, I have decided to

- Impute the missing Age values from a normal distribution
- Scale the Age values
- Impute the missing Embarked values with the most frequent value.
- Imputing the missing Fare values with mean.
- Checking for outliers in Fare values and then performing RobustScaler and StandardScaler.
- Drop Ticket [many duplicates]
- Extracting Deck information from Cabin column and label encoding them.
- Extracting the 'Title' from PassengerName column and grouping them as Dr, Mr, Mrs, Miss, Master and Others and label encoding them.
- Dropping the PassengerID column as it gives no predictive information.
- Extracted information regarding the number of relatives and whether the passengers are traveling alone or not from the given information regarding parents and children of passengers as this seemed like an important feature for prediction.

I also performed some feature engineering as there are few categorical variables present. Performed Label Encoding on:

- Sex
- Embarked
- Title

I have also used feature_importances_ function from Random Forest Classifier to check for the most important features along with representative visualizations.

![Representation of important features](https://www.kaggleusercontent.com/kf/88770047/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Yy9hiYhCCqDnyhmz8wKeDQ.q0gzVKWKm2N1S3t8yRkDf1YVhpqSjR7e6Cja8a-U5Isd-c8O3GxKmJ7VN1eVXSwMDIQ4IlVpGHbUCh5EQJ6tWos6iC1G5txp2M5c8Pgt3GQ0gD3NXouXigl78szEquNz1m1vBnsUJsTYE8Uikwa-ZGPE4IiK4cz3IrNnEP7aOBu81I9lZ4yt9msRP2JnzrzQ22gu5FIJtMotZw6dIWw5_rsaw40TYGoKDng9PInLU8NyKU7aFNjOrA0uGRF4xEYVuSWDTjqXjisyf4UIYg79nuMgdwRMM0EvlxabI00S4_LweQ3PT8VnEGE7MaIlRIndiTzbtsTMPhkjmuANi5-JFoSfQ1Gnl5BT6nvMGM3retpshXCxrRw2rRb5jAhM2ydyOmff9C-1p46rCfDwRHmmaXKvUYyIBvFiaUmk22S9c7Ej861kwuvH_zhRZu996T5kMjrxClStSKPPNLH32R7_JOAv7-oJ5lkbIAtfo0tvtl-G1TaTXAradNEJ-irfxTQSmOSagyq-29dclXLOVGG3f32I7pDIZQvEuLEe-Xzx3GuUVf4HSWOSzpzti9Nwggh8Zac24bN0TYEVBqStF0LU1z6Vf3phVkH4ovB75fkQKo0DaYCJosZS3_ExF3XNz6bPwgDlC28JWiWSFcgSKiIcAWyUOkzA-EMV6bGx0eC3RTg.pgA7_lxbYpQd4F1PSc5VLQ/__results___files/__results___67_0.png)

## Modeling

Models trained:

- Random Forest
- Logistic Regression
- Guassian Naive Bayes
- K-Nearest Neighbors
- Decision Tree

## Evaluation

Random Forest Classifier performed the best out of the lot with **84%** accuracy on the training set and 77.75% accuracy on the test set.
The consistent results in accuracy proves that the models did not experience underfitting or overfitting.
