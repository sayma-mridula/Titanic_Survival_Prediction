# 🛳️ Titanic Survival Prediction

This project uses the classic Titanic dataset to build a machine learning model that predicts whether a passenger survived or not based on features like age, sex, passenger class, and more.

## 📂 Files Included

- `Titanic_Survival_Prediction.ipynb`: Main Jupyter Notebook containing data preprocessing, visualization, and modeling.
- `train.csv`: Dataset containing information about Titanic passengers used for training the model.

## 📊 Dataset Overview

The dataset (`train.csv`) includes:
- `PassengerId`: Unique ID for each passenger
- `Survived`: Target variable (0 = No, 1 = Yes)
- `Pclass`: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Name`, `Sex`, `Age`: Personal information
- `SibSp`, `Parch`: Family onboard
- `Ticket`, `Fare`: Ticket details
- `Cabin`: Cabin number (many missing)
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## ✅ Steps Performed

1. **Data Cleaning**
   - Handled missing values in `Age`, `Embarked`, etc.
   - Converted categorical variables like `Sex` and `Embarked` into numeric form

2. **Exploratory Data Analysis (EDA)**
   - Plotted survival distribution based on age, sex, class
   - Visualized missing data patterns

3. **Feature Engineering**
   - Created new features or dropped less useful ones (if any)

4. **Modeling**
   - Used `LogisticRegression` from `scikit-learn` to train a binary classifier
   - Evaluated using accuracy score

5. **Prediction**
   - Model is trained to predict survival on test inputs

## 🧪 Example Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
