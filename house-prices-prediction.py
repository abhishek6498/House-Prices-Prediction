# %% [code]
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# # Stage 1. Data acquisition

# %% [code]
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# Import the file train.csv and save it in a variable train_df

# %% [code]
train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv", index_col="Id")

# %% [code]
train_df.head()

# %% [markdown]
# General information

# %% [code]
train_df.info()

# %% [markdown]
# Let's find missing values

# %% [code]
train_df.isnull().sum()[train_df.isnull().sum() > 0]

# %% [markdown]
# Let's generate descriptive statistics.

# %% [code]
train_df.describe(include=["int64", "float64"])

# %% [markdown]
# # Stage 2. EDA and Data preprocessing

# %% [markdown]
# Let's see the correlation of parameters on the heat map

# %% [code]
# following are initialized for later use in notebook
quantitative_train_df = train_df.select_dtypes(include=["int64", "float64"])
quantitative_variables = list(quantitative_train_df.columns)
quantitative_variables.remove("SalePrice")
qualitative_train_df = train_df.select_dtypes(include=["object"])
qualitative_variables = list(qualitative_train_df.columns)

# %% [code]
plt.figure(figsize=(14,7))
sns.heatmap(data=quantitative_train_df.corr(), annot=True, annot_kws={"size": 6},cmap='coolwarm', linewidths=.5)

# %% [markdown]
# On the heat map, we see that the **OverallQual**, **GrLivArea** columns has the highest correlation with the **SalePrice** column.

# %% [code]
sns.displot(data=quantitative_train_df, x="SalePrice", kde=True)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Sale Prices")
plt.ylabel("Frequency")
plt.title("Distribution of Sale Prices")

# %% [markdown]
# **Conclusion**: 
# 
# The chart shows that most prices are in the region of 100 000-125 000.

# %% [code]
sns.jointplot(data=quantitative_train_df, x="GrLivArea", y="SalePrice")

# %% [markdown]
# **Conclusion**:
# 
# The graph shows that most buildings are between 750 and 2 000 square feet.

# %% [code]
B_S = sns.catplot(x='BldgType', y='SalePrice', data=train_df, kind='bar')
for ax in B_S.axes.flat:
    for p in ax.patches:
        x_coord = p.get_x() + 0.5 * p.get_width()
        value = p.get_height()
        ax.annotate(f'{value:.0f}', (x_coord, value), ha='left')
plt.xlabel('Building Type (BldgType)')
plt.ylabel('Mean Sale Price')
plt.title('Mean Sale Price by Building Type')

# %% [markdown]
# **Conclusion:**
# 
# The graph shows that the highest average price for buildings of the type **1Fam** and **TwnhsE**.

# %% [code]
M = sns.countplot(data=train_df, x="MSZoning")
for p in M.patches:
    x_coord = p.get_x() + 0.5 * p.get_width()
    value = p.get_height()
    M.annotate(f'{value:.0f}', (x_coord, value), ha='center', va='bottom')
plt.xlabel('Zoning Type')
plt.ylabel('Count of properties')
plt.title('Distribution of Zoning Types')

# %% [markdown]
# **Conclusion:**
# 
# The graph shows that a significantly larger number of buildings with zoning type **RL**.

# %% [code]
plt.figure(figsize=(15,6))
sns.boxplot(data=train_df, x="Neighborhood", y="SalePrice")
plt.xlabel("Neighborhood")
plt.ylabel("SalePrice")
plt.xticks(rotation=45)
plt.title("Neighborhood vs SalePrice")

# %% [markdown]
# **Conclusion:**
# 
# The graph shows that the largest price fork is observed in neighborhoods **NridgHt** and **StoneBr**

# %% [markdown]
# By looking at Doument.txt, all of categorical(Except CentralAir) seems to be of ordinal type. Hence, I will use Label Encoding to encode categorical variables.

# %% [code]
train_df['MSZoning'] = train_df['MSZoning'].replace({'RM': 1, 'RP': 2, 'RL': 3, 'RH': 4, 'I': 5, 'FV': 6, 'C': 7, 'A': 8, 'C (all)': np.nan})

train_df['Street'] = train_df['Street'].replace({'Pave': 1, 'Grvl': 2})

train_df['Alley'] = train_df['Alley'].replace({'NA': 1, 'Pave': 2, 'Grvl': 3})

train_df['LotShape'] = train_df['LotShape'].replace({'IR3': 1,'IR2': 2, 'IR1': 3, 'Reg': 4})

train_df['LandContour'] = train_df['LandContour'].replace({'Low': 1,'HLS': 2, 'Bnk': 3, 'Lvl': 4})

train_df['Utilities'] = train_df['Utilities'].replace({'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4})

train_df['LotConfig'] = train_df['LotConfig'].replace({'FR3': 1, 'FR2': 2, 'CulDSac': 3, 'Corner': 4, 'Inside': 5})

train_df['LandSlope'] = train_df['LandSlope'].replace({'Gtl': 1, 'Mod': 2, 'Sev': 3})

train_df['Neighborhood'] = train_df['Neighborhood'].replace({'Veenker': 0,'Timber': 1, 'StoneBr': 2, 'Somerst': 3, 'SawyerW': 4, 'Sawyer': 5,
                                                           'SWISU': 6,'OldTown': 7, 'NWAmes': 8, 'NridgHt': 9, 'NPkVill': 10, 'NoRidge': 11,
                                                           'Names': 12,'Mitchel': 13, 'MeadowV': 14, 'IDOTRR': 15, 'Gilbert': 16, 'Edwards': 17,
                                                           'Crawfor': 18,'CollgCr': 19, 'ClearCr': 20, 'BrkSide': 21, 'BrDale': 22, 'Blueste': 23,
                                                           'Blmngtn': 24, 'NAmes': 12})

train_df['Condition1'] = train_df['Condition1'].replace({'RRAe': 1,'RRNe':2, 'PosA': 3, 'PosN': 4,'RRAn': 5,
                                                           'RRNn': 6,'Norm': 7,'Feedr': 8, 'Artery': 9})

train_df['Condition2'] = train_df['Condition2'].replace({'RRAe': 1,'RRNe':2, 'PosA': 3, 'PosN': 4,'RRAn': 5,
                                                           'RRNn': 6,'Norm': 7,'Feedr': 8, 'Artery': 9})

train_df['BldgType'] = train_df['BldgType'].replace({'TwnhsI': 1, 'TwnhsE': 2, 'Duplx': 3, '2FmCon': 4, '1Fam': 5, 'Duplex': 3, '2fmCon': 4, 
                                                     'Twnhs': 2})

train_df['HouseStyle'] = train_df['HouseStyle'].replace({'SLvl': 1, 'SFoyer': 2, '2.5Unf': 3, '2.5Fin': 4,'2Story': 5,
                                                           '1.5Unf': 6,'1.5Fin': 7,'1Story': 8})

train_df['RoofStyle'] = train_df['RoofStyle'].replace({'Shed': 1, 'Mansard': 2, 'Hip': 3, 'Gambrel': 4, 'Gable' : 5, 'Flat': 6})

train_df['RoofMatl'] = train_df['RoofMatl'].replace({'ClyTile': 8,'CompShg': 7, 'Membran': 6, 'Metal': 5,'Roll': 4,
                                                     'Tar&Grv': 3,'WdShake': 2,'WdShngl': 1})

train_df['Exterior1st'] = train_df['Exterior1st'].replace({'WdShing': 1, 'Wd Sdng': 2, 'VinylSd': 3, 'Stucco': 4, 'Stone': 5,
                                                           'PreCast': 6,'Plywood': 7, 'Other': 8, 'MetalSd': 9, 'ImStucc': 10,
                                                           'HdBoard': 11,'CemntBd': 12,'CBlock': 13, 'BrkFace': 14, 'BrkComm': 15,
                                                           'AsphShn': 16, 'AsbShng': 17})

train_df['Exterior2nd'] = train_df['Exterior2nd'].replace({'VinylSd': 1, 'Wd Sdng': 2, 'HdBoard': 3, 'Plywood': 4, 'MetalSd': 5, 'Brk Cmn': 6,
                                                         'CmentBd': 7, 'ImStucc': 8, 'Wd Shng': 9, 'AsbShng': 10, 'Stucco': 11, 'CBlock': 12,
                                                         'BrkFace': 13, 'AsphShn': 14,'Stone': 15,'Other': 16})

# %% [code]
train_df['MasVnrType'] = train_df['MasVnrType'].replace({'Stone': 1, 'None': 2, 'CBlock': 3, 'BrkFace': 4, 'BrkCmn' : 5})

train_df['ExterQual'] = train_df['ExterQual'].replace({'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4,'Ex' : 5})

train_df['ExterCond'] = train_df['ExterCond'].replace({'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4,'Ex' : 5})

train_df['Foundation'] = train_df['Foundation'].replace({'Wood' : 1, 'Stone' : 2, 'Slab' : 3, 'PConc' : 4, 'CBlock' : 5, 'BrkTil' : 6})

train_df['BsmtQual'] = train_df['BsmtQual'].replace({'NA' : 1, 'Po' : 2, 'Fa': 3, 'TA' : 4, 'Gd' : 5, 'Ex' : 6})

train_df['BsmtCond'] = train_df['BsmtCond'].replace({'NA' : 1, 'Po' : 2, 'Fa': 3, 'TA' : 4, 'Gd' : 5, 'Ex' : 6})

train_df['BsmtExposure'] = train_df['BsmtExposure'].replace({'NA' : 1, 'No' : 2, 'Mn': 3, 'Av' : 4, 'Gd' : 5})

train_df['BsmtFinType1'] = train_df['BsmtFinType1'].replace({'NA' : 1, 'Unf' : 2, 'LwQ': 3, 'Rec' : 4, 'BLQ' : 5, 'ALQ' : 6, 'GLQ' : 7})

train_df['BsmtFinType2'] = train_df['BsmtFinType2'].replace({'NA' : 1, 'Unf' : 2, 'LwQ': 3, 'Rec' : 4, 'BLQ' : 5, 'ALQ' : 6, 'GLQ' : 7})

train_df['Heating'] = train_df['Heating'].replace({'Wall' : 1, 'OthW' : 2, 'Grav': 3, 'GasW' : 4, 'GasA' : 5, 'Floor' : 6})

train_df['HeatingQC'] = train_df['HeatingQC'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4,'Ex': 5})

train_df['CentralAir'] = train_df['CentralAir'].replace({'Y': 0, 'N': 1})

train_df['Electrical'] = train_df['Electrical'].replace({'Mix': 1, 'FuseF': 2, 'FuseA': 3, 'FuseP': 4, 'SBrkr': 5})

train_df['KitchenQual'] = train_df['KitchenQual'].replace({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

train_df['Functional'] = train_df['Functional'].replace({'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ' : 8})

# %% [code]
train_df['FireplaceQu'] = train_df['FireplaceQu'].replace({'NA' : 1, 'Po' : 2, 'Fa': 3, 'TA' : 4, 'Gd' : 5, 'Ex' : 6})

train_df['GarageType'] = train_df['GarageType'].replace({'NA' : 1, 'Detchd' : 2, 'CarPort' : 3, 'BuiltIn' : 4, 'Basment' : 5, 'Attchd': 6, '2Types' : 7})

train_df['GarageFinish'] = train_df['GarageFinish'].replace({'NA' : 1, 'Unf' : 2, 'RFn' : 3, 'Fin' : 4})

train_df['GarageQual'] = train_df['GarageQual'].replace({'NA' : 1, 'Po' : 2, 'Fa': 3, 'TA' : 4, 'Gd' : 5, 'Ex' : 6})

train_df['GarageCond'] = train_df['GarageCond'].replace({'NA' : 1, 'Po' : 2, 'Fa': 3, 'TA' : 4, 'Gd' : 5, 'Ex' : 6})

train_df['PavedDrive'] = train_df['PavedDrive'].replace({'N' : 1, 'P' : 2, 'Y' : 3})

train_df['PoolQC'] = train_df['PoolQC'].replace({'NA' : 1, 'Fa': 3, 'TA' : 4, 'Gd' : 5, 'Ex' : 6})

train_df['Fence'] = train_df['Fence'].replace({'NA' : 1, 'MnWw' : 2, 'GdWo': 3, 'MnPrv' : 4, 'GdPrv' : 5})

train_df['MiscFeature'] = train_df['MiscFeature'].replace({'NA' : 1, 'TenC' : 2, 'Shed' : 3, 'Othr' : 4, 'Gar2' : 5, 'Elev' : 6})

train_df['SaleType'] = train_df['SaleType'].replace({'Oth' : 1, 'ConLD' : 2, 'ConLI' : 3, 'ConLw' : 4, 'Con' : 5, 'COD' : 6,
                                                     'New' : 7, 'VWD' : 8, 'CWD' : 9, 'WD' : 10})

train_df['SaleCondition'] = train_df['SaleCondition'].replace({'Partial' : 1, 'Family' : 2, 'Alloca' : 3, 'AdjLand' : 4, 'Abnorml' : 5, 'Normal' : 6})

# %% [markdown]
# Let's fill in the missing values: Gonna drop the columns with more than 25% missing values, and use KNNImputer to impute missing values in the rest of columns 

# %% [code]
percentage_missing = train_df.isnull().sum()[train_df.isnull().sum()>0]/len(train_df)*100

# %% [code]
percentage_missing[percentage_missing > 25]

# %% [code]
dropped_columns = ['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
train_df.drop(columns=dropped_columns, inplace=True)

# %% [code]
from sklearn.impute import KNNImputer
knn_imputer = KNNImputer(n_neighbors=1)
knn_imputed_dataset = knn_imputer.fit_transform(train_df)
knn_imputed_dataset_df = pd.DataFrame(data=knn_imputed_dataset, columns=list(train_df.columns), index=list(train_df.index))
knn_imputed_dataset_df.info()

# %% [markdown]
# let's prepare the training and testing dataset:

# %% [code]
X = knn_imputed_dataset_df.drop(columns=["SalePrice"])
y = knn_imputed_dataset_df["SalePrice"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %% [markdown]
# # Stage 3. Machine learning models

# %% [markdown]
# I am going to try following models:
# 
# 1. regularised linear regression
# 
# 2. XGBoost
# 
# 3. RandomForest
# 
# 4. SVR
# 
# 5. Neural Networks

# %% [markdown]
# define model evaluation method:

# %% [code]
from sklearn.metrics import r2_score
r2scores = {"lasso_linear_regression": 0, "XGBoost": 0, "random_forest": 0, "svm": 0, "neural_networks": 0}

# %% [markdown]
# # 3.1 regularised linear regression

# %% [markdown]
# since linear regression does better without outliers, I am going to remove potential outliers.

# %% [code]
X_train_linear_regression = X_train
for column in list(X_train_linear_regression.columns):
    Q1 = X_train_linear_regression[column].quantile(0.25)
    Q3 = X_train_linear_regression[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    X_train_linear_regression[column] = np.where(X_train_linear_regression[column] < lower_bound, lower_bound, 
                                                       X_train_linear_regression[column])
    X_train_linear_regression[column] = np.where(X_train_linear_regression[column] > upper_bound, upper_bound, 
                                                          X_train_linear_regression[column])

# %% [code]
from sklearn.preprocessing import StandardScaler

standard_scaler_X = StandardScaler().fit(X_train_linear_regression)
X_train_lr_scaled = standard_scaler_X.transform(X_train_linear_regression)
X_test_scaled = standard_scaler_X.transform(X_test)

# %% [code]
from sklearn.linear_model import Lasso
lasso_linear_regression = Lasso(alpha=0.1)
lasso_linear_regression.fit(X_train_lr_scaled, y_train)

# %% [markdown]
# Let's predict the Sale prices for test dataset

# %% [code]
y_pred_llr = lasso_linear_regression.predict(X_test_scaled)

# %% [code]
plt.scatter(y_test, y_pred_llr)
plt.plot([0,max(y_test)], [0, max(y_test)], "go--")
plt.xlabel("Actual Sales Prices")
plt.ylabel("Predicted Sales Prices")
plt.title("Actual vs. Predicted  Sales Prices for regularised linear regression model")
plt.show()

# %% [markdown]
# most of scatter plot seems to be close to Line-of-Perfect-Prediction(green line, where predictions=actuals), means regularised linear regression seems to be perfoming well. 

# %% [code]
r2scores["lasso_linear_regression"] = r2_score(y_test, y_pred_llr)
print(r2scores["lasso_linear_regression"])

# %% [markdown] {"execution":{"iopub.status.busy":"2023-09-17T15:09:03.313322Z","iopub.execute_input":"2023-09-17T15:09:03.313806Z","iopub.status.idle":"2023-09-17T15:09:03.321666Z","shell.execute_reply.started":"2023-09-17T15:09:03.313769Z","shell.execute_reply":"2023-09-17T15:09:03.320061Z"}}
# Implies that regularised linear model has explained 87.1% of variance in test data

# %% [markdown]
# # 3.2 XGBoost

# %% [markdown]
# install xgboost:

# %% [code]
!pip install xgboost

# %% [markdown]
# import XGBRegressor:

# %% [code]
import xgboost

# %% [code]
from xgboost import XGBRegressor

# %% [code]
xgboost_model = XGBRegressor(random_state=42)

# %% [markdown]
# Let's tune the hyperparameters:

# %% [code]
from sklearn.model_selection import GridSearchCV

# %% [code]
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

# %% [code]
grid_search = GridSearchCV(estimator=xgboost_model, param_grid=param_grid, scoring='r2', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# %% [code]
best_params = grid_search.best_params_
print(best_params)

# %% [code]
xgboost_model_tuned = grid_search.best_estimator_

# %% [markdown]
# Let's predict Sale prices for test dataset using tuned XGBRegressor model:

# %% [code]
y_pred_xgboost = xgboost_model_tuned.predict(X_test)

# %% [code]
plt.scatter(y_test, y_pred_xgboost)
plt.plot([0,max(y_test)], [0, max(y_test)], "go--")
plt.xlabel("Actual Sales Prices")
plt.ylabel("Predicted Sales Prices")
plt.title("Actual vs. Predicted  Sales Prices for xgboost regression model")
plt.show()

# %% [markdown]
# most of scatter plot seems to be close to Line-of-Perfect-Prediction(green line, where predictions=actuals), means XGBoost regression seems to be perfoming well. 

# %% [code]
r2scores["XGBoost"] = r2_score(y_test, y_pred_xgboost)
print(r2scores["XGBoost"])

# %% [markdown]
# Implies that xgboost regression has explained 90% of variance in test data.

# %% [markdown]
# # 3.3 Random Forest

# %% [code]
from sklearn.ensemble import RandomForestRegressor

# %% [code]
random_forest_model = RandomForestRegressor(oob_score=True)

# %% [code]
param_grid = {
    'n_estimators': [301, 501],
    'max_features': [4, 12, 31, 36, 54, 73],
    'max_depth': [7, 10],
    'min_samples_leaf': [25, 50, 100],
    'min_samples_split': [75, 150, 300]
}

# %% [code]
grid_search_rf = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, scoring='r2', cv=5, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# %% [code]
grid_search_rf.best_params_

# %% [code]
random_forest_model_tuned = grid_search_rf.best_estimator_

# %% [markdown]
# Let's predict Sale Prices for test data using tuned Random forest regression model

# %% [code]
y_pred_rf = random_forest_model_tuned.predict(X_test)

# %% [code]
plt.scatter(y_test, y_pred_rf)
plt.plot([0,max(y_test)], [0, max(y_test)], "go--")
plt.xlabel("Actual Sales Prices")
plt.ylabel("Predicted Sales Prices")
plt.title("Actual vs. Predicted  Sales Prices for random forest regression model")
plt.show()

# %% [markdown]
# most of scatter plot seems to be close to Line-of-Perfect-Prediction(green line, where predictions=actuals), means Random Forest regression seems to be perfoming well. 

# %% [code]
r2scores["random_forest"] = r2_score(y_test, y_pred_rf)
print(r2scores["random_forest"])

# %% [markdown]
# Implies that random forest regression model has explained 79.8% of variance in test data.

# %% [markdown]
# # 3.4 SVM

# %% [code]
from sklearn.svm import SVR

# %% [markdown]
# since SVM is also prone to outlieres and does better with normalised data, I am going to use same X_train_scaled and X_test_scaled for SVM as used for regularised linear regression

# %% [code]
svr_model = SVR()
svr_model.fit(X=X_train_lr_scaled, y=y_train)

# %% [markdown]
# Let's predict Sale Prices for test data using SVM regression model

# %% [code]
y_pred_svr = svr_model.predict(X_test_scaled)

# %% [code]
plt.scatter(y_test, y_pred_svr)
plt.plot([0,max(y_test)], [0, max(y_test)], "go--")
plt.xlabel("Actual Sales Prices")
plt.ylabel("Predicted Sales Prices")
plt.title("Actual vs. Predicted  Sales Prices for svm regression model")
plt.show()

# %% [markdown]
# most of scatter plot doesn't seem to be close to Line-of-Perfect-Prediction(green line, where predictions=actuals), means SVM regression has performed poorly. 

# %% [code]
r2scores["svm"] = r2_score(y_test, y_pred_svr)
print(r2scores["svm"])

# %% [markdown]
# Implies that SVM regression performed very poorly as R2 score is negative.

# %% [markdown]
# # 3.5 Neural Networks

# %% [code]
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# %% [markdown]
# Since neural networks does better with normalised data, I am going to tranform X_train and X_test with StandardScaler

# %% [code]
scaler_nn_X = StandardScaler().fit(X_train)
X_train_nn_scaled = scaler_nn_X.transform(X_train)
X_test_nn_scaled = scaler_nn_X.transform(X_test)

y_train_reshaped = y_train.to_numpy().reshape(-1, 1)
scaler_nn_y = StandardScaler().fit(y_train_reshaped)
y_train_nn_scaled = scaler_nn_y.transform(y_train_reshaped)

y_test_reshaped = y_test.to_numpy().reshape(-1, 1)
y_test_nn_scaled = scaler_nn_y.transform(y_test_reshaped)

# %% [code]
nn_model = Sequential()
nn_model.add(Dense(32, input_shape=(X_train_nn_scaled.shape[1],), activation='relu'))
nn_model.add(Dropout(0.4))
nn_model.add(Dense(16, activation='relu'))
nn_model.add(Dropout(0.4))
nn_model.add(Dense(1))

# %% [code]
nn_model.compile(loss='mse', optimizer='sgd', metrics=['mse'])

# %% [code]
nn_model.fit(X_train_nn_scaled, y_train_nn_scaled, epochs=5, batch_size=32, validation_split=0.2)

# %% [markdown]
# Let's predict Sale prices for test data using Neural Networks model:

# %% [code]
y_pred_nn_scaled = nn_model(X_test_nn_scaled)
y_pred_nn_scaled = y_pred_nn_scaled.numpy().reshape(-1, 1)
y_pred_nn_unscaled = scaler_nn_y.inverse_transform(y_pred_nn_scaled)

# %% [code]
plt.scatter(y_test, y_pred_nn_unscaled)
plt.plot([0,max(y_test)], [0, max(y_test)], "go--")
plt.xlabel("Actual Sales Prices")
plt.ylabel("Predicted Sales Prices")
plt.title("Actual vs. Predicted  Sales Prices for neural networks regression model")
plt.show()

# %% [markdown]
# most of scatter plot seems to be close to Line-of-Perfect-Prediction(green line, where predictions=actuals), but most of predicted Sale Prices are less than 0.25. Clearly our Neural Networks model is overfitting on redundant feature like "Sale Prices must be close to 0"

# %% [code]
r2scores["neural_networks"] = r2_score(y_test, y_pred_nn_unscaled)
print(r2scores["neural_networks"])

# %% [markdown]
# Implies that Neural networs performed poorly as R2 score is negative.

# %% [code]
print("R2 score for regularised linear regression model: ", r2scores["lasso_linear_regression"])
print("R2 score for XGBoost regression model: ", r2scores["XGBoost"])
print("R2 score for random forest regression model: ", r2scores["random_forest"])
print("R2 score for svm regression model: ", r2scores["svm"])
print("R2 score for neural networks regression model: ", r2scores["neural_networks"])

# %% [markdown]
# # Stage 4: Aggregation of predictions

# %% [markdown]
# since relularised linear regression model and XGBoost model has highest R2 scores, I am going to combine thier predictions by taking average, proceed with final predictions

# %% [code]
y_pred_final = (y_pred_llr + y_pred_xgboost)/2

# %% [code]
plt.scatter(y_test, y_pred_final)
plt.plot([0,max(y_test)], [0, max(y_test)], "go--")
plt.xlabel("Actual Sales Prices")
plt.ylabel("Predicted Sales Prices")
plt.title("Actual vs. Predicted  Sales Prices for combined regression model")
plt.show()

# %% [code]
print(r2_score(y_test, y_pred_final))