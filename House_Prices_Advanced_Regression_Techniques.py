
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from sklearn.preprocessing import LabelEncoder  ###for encode a categorical values
from sklearn.model_selection import train_test_split  ## for spliting the data
# from lightgbm import LGBMRegressor    ## for import our model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from scipy.stats import norm, skew #for some statistics



# In[2]:


train_dataset = pd.read_csv('G:\\sliit DS\\4th year 1st seme\\thawes kaggle\\house-prices-advanced-regression-techniques\\train.csv')
test_dataset = pd.read_csv('G:\\sliit DS\\4th year 1st seme\\thawes kaggle\\house-prices-advanced-regression-techniques\\test.csv')


# In[3]:


train_dataset.head()


# In[4]:


train_dataset.shape


# In[5]:


train_dataset.describe()


# In[6]:


test_dataset.shape


# In[7]:


test_dataset.head()


# In[8]:


train_dataset.columns


# In[9]:


test_dataset.columns

Feature Engineering
# In[10]:


train_dataset.dtypes


# In[11]:


#change salesPrice int64 type to float64
train_dataset["SalePrice"]=train_dataset["SalePrice"].astype("float64")


# In[12]:


test_dataset.dtypes


# In[13]:


#Save the 'Id' column
train_ID = train_dataset['Id']
test_ID = test_dataset['Id']


# In[14]:


#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train_dataset.drop("Id", axis = 1, inplace = True)
test_dataset.drop("Id", axis = 1, inplace = True)


# In[15]:


#Check for null values
train_dataset.isnull().sum()


# In[16]:


#drop major missing value columns in train_dataset
train_dataset.drop("Alley", axis = 1, inplace = True)
train_dataset.drop("PoolQC", axis = 1, inplace = True)
train_dataset.drop("Fence", axis = 1, inplace = True)
train_dataset.drop("MiscFeature", axis = 1, inplace = True)


# In[17]:


train_dataset.isnull().sum()


# In[18]:


# We have fill numerical columns misssing values with median and We will fill character missing values with most used value count

col_miss_val_train = [col for col in train_dataset.columns if train_dataset[col].isnull().any()]
print(col_miss_val_train)

for col in col_miss_val_train:
    if(train_dataset[col].dtype == np.dtype('O')):
         train_dataset[col]=train_dataset[col].fillna(train_dataset[col].value_counts().index[0])    #replace NaN values with most frequent value
    else:
        train_dataset[col] = train_dataset[col].fillna(train_dataset[col].median()) 
        


# In[19]:


test_dataset.isnull().sum()


# In[20]:


#drop major missing value columns in test_dataset
test_dataset.drop("Alley", axis = 1, inplace = True)
test_dataset.drop("PoolQC", axis = 1, inplace = True)
test_dataset.drop("Fence", axis = 1, inplace = True)
test_dataset.drop("MiscFeature", axis = 1, inplace = True)


# In[21]:


test_dataset.isnull().sum()


# In[22]:


# We have fill numerical columns misssing values with median and We will fill character missing values with most used value count

col_miss_val_test = [col for col in test_dataset.columns if test_dataset[col].isnull().any()]
print(col_miss_val_test)

for col in col_miss_val_test:
    if(test_dataset[col].dtype == np.dtype('O')):
         test_dataset[col]=test_dataset[col].fillna(test_dataset[col].value_counts().index[0])    #replace NaN values with most frequent value
    else:
        test_dataset[col] = test_dataset[col].fillna(test_dataset[col].median()) 
        


# In[23]:


#Outliers
fig, ax = plt.subplots()
ax.scatter(x = train_dataset['GrLivArea'], y = train_dataset['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[24]:


#We can see at the bottom right two with extremely large GrLivArea that are of a low price. These values are huge oultliers. Therefore, we can safely delete them.
#Deleting outliers
train_dataset = train_dataset.drop(train_dataset[(train_dataset['GrLivArea']>4000) & (train_dataset['SalePrice']<300000)].index)


# In[25]:


#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(x = train_dataset['GrLivArea'], y = train_dataset['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[26]:


#Coding categorical value into numerical value.Label encoder
train_dataset.select_dtypes(include=['object'])


LE = LabelEncoder()
for col in train_dataset.select_dtypes(include=['object']):
    train_dataset[col] = LE.fit_transform(train_dataset[col])
    
train_dataset.head()


# In[27]:


test_dataset.select_dtypes(include=['object'])


LE = LabelEncoder()
for col in test_dataset.select_dtypes(include=['object']):
    test_dataset[col] = LE.fit_transform(test_dataset[col])
    
test_dataset.head()


# Target Variable

# In[28]:


#Check for normal distribution for y variable.Models love normal distribution.
sns.distplot(train_dataset['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_dataset['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))


# In[29]:


#Get also the QQ-plot
from scipy import stats

fig = plt.figure()
res = stats.probplot(train_dataset['SalePrice'], plot=plt)
plt.show()


# In[30]:


#The target variable is right skewed. As (linear) models love normally distributed data ,
#we need to transform this variable and make it more normally distributed

# Log-transformation of the target variable
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train_dataset["SalePrice"] = np.log1p(train_dataset["SalePrice"])

#Check the new distribution 
sns.distplot(train_dataset['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_dataset['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_dataset['SalePrice'], plot=plt)
plt.show()


# In[31]:


#Data correlation
#Correlation map to see how features are correlated with SalePrice
corrmat = train_dataset.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# Predictive model building

# In[32]:


#Split train_dataset
x = train_dataset.iloc[:,0:-1]
y = train_dataset.iloc[:,-1] 


# In[33]:


#split train,test
x_train , x_test , y_train , y_test = train_test_split(x , y ,test_size = 0.1,random_state = 1)


# In[34]:


x_train


# In[35]:


#Import libraries
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# In[36]:


#Base models

#LASSO Regression
# This model may be very sensitive to outliers. So we need to made it more robust on them.
# For that we use the sklearn's Robustscaler() method on pipeline
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

# Elastic Net Regression 
# again made robust to outliers
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

# Kernel Ridge Regression 
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

# Gradient Boosting Regression :
# With huber loss that makes it robust to outliers

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

# XGBoost :
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

# LightGBM :
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# =>  Select Best models

# In[37]:


#Base model scores
#We use cross validation score for neg_mean_squared_error 
#Validation function
# If y variable is continuous variable we use RMSE Measure. Not accuracy measure.accuracy score is used in classification problem
n_folds = 5

# RMSLE (Root Mean Square Logaithmic Error)
#This method is used for use RMSE meseare inside cross validation method
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train.values)
    rmse= np.sqrt(-cross_val_score(model, x_train.values, y_train.values, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[38]:


score = rmsle_cv(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[39]:


score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[40]:


# score = rmsle_cv(KRR)
# print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[41]:


score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[42]:


score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[43]:


score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# In[44]:


#Average base models score
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   
    
averaged_models = AveragingModels(models = (ENet, GBoost, lasso, model_xgb, model_lgb))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# =>  Fit the model to test_dataset and make predictions

# In[45]:


# Ensemble Technique
# Stacking


# In[46]:


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


# In[47]:


# meta model is used to Use the predictions from 3) (called out-of-folds predictions) as the inputs, and the correct responses
# (target variable) as the outputs to train a higher level.
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, lasso, model_lgb), meta_model = model_xgb)


# In[48]:


# RMSLE(Root Mean Square logarithmic Error)
#This method is for calculate RMSE between predicted y values and actual y values
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[49]:


# Stacked Regressor 
stacked_averaged_models.fit(x_train.values, y_train.values)
stacked_x_test_pred = stacked_averaged_models.predict(x_test.values)
print("RMSLE Score for Stacked Regressor :{}".format(rmsle(y_test.values, stacked_x_test_pred)))


# In[50]:


# XGBoost
model_xgb.fit(x_train, y_train)
xgb_x_test_pred = model_xgb.predict(x_test)
print("RMSLE Score for XGBoost :{}".format(rmsle(y_test, xgb_x_test_pred)))


# In[51]:


# GradientBoost:
GBoost.fit(x_train, y_train)
gb_x_test_pred = GBoost.predict(x_test)
print("RMSLE Score for GradientBoost :{}".format(rmsle(y_test, gb_x_test_pred)))


# In[52]:


stacked_test_dataset_pred = np.expm1(stacked_averaged_models.predict(test_dataset))  #np.expm1() means [(Exponential value of array element) - (1)].
                                                                                            # This mathematical function helps user to calculate exponential of all the elements subtracting 1 from all the input array elements.
xgb_test_dataset_pred = np.expm1(model_xgb.predict(test_dataset))
gb_test_dataset_pred = np.expm1(GBoost.predict(test_dataset))


# In[53]:


'''RMSE on the entire Train data when averaging'''

print('RMSLE score on x_test data:')
print(rmsle(y_test,stacked_x_test_pred*0.7 +
               xgb_x_test_pred*0.15 + gb_x_test_pred*0.15 ))


# In[56]:


#Weigted Average
# This is an extension of the averaging method. 
# All models are assigned different weights defining the importance of each model for prediction. 

#weights have been chosen where rmse value is low on x_test data
#Esemble prediction
ensemble = stacked_test_dataset_pred*0.7 + xgb_test_dataset_pred*0.15 + gb_test_dataset_pred*0.15
ensemble


# In[57]:


sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)

