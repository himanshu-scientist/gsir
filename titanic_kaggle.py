"""Lets create kaggle ML project  ---  kaggle.com/c/titanic   ---  """
"""Problem Statement- Predict survival on Titanic"""

# basic 4 steps. 1) Explanatory data analysis 2) Feature engineering 3) Cross validation 4) hyper parameter tunning.
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
input_file=pd.read_csv(r"D:\Study\titanic\train.csv")
input_file2=pd.read_csv(r"D:\Study\titanic\test.csv")
train_data=pd.DataFrame(input_file)
"""Exploratory Data Analysis (EDA)
EDA is a very useful approach which is in line with the statement that "A picture is worth a thousand words". If done properly,
 it can reveal many secrets about our data, 
and it is a very important tool for a competitive Kaggler. So, why it is a good idea to do EDA? Well, some good reasons are the following:
Explore the size of the data. Obviously, a 100 observations times 10 variables dataset is a totally different story than
 a 1000000 observations times 100 variables dataset.
Explore the target variable properties.
Explore features properties.
Explore patterns in the data to help us for feature engineering.
OK, so let's start our EDA! Let's try to follow our previous logic, step by step:"""


"""Feature engineering is the transformation of data into features, with the aim of 
representing better the problem we would like to solve, and resulting to a better model performance on unseen data"""
# print(train_data.shape) # 891,12-- means there are 891 rows and 12 columns
pd.set_option('display.max_columns',12)# I selected 12 columns to see at 1 time
# print(train_data.info())
# print(train_data.head(5))
# print(train_data.isnull().sum().sort_values(ascending=False)) # we can see that there are total 687 null
# values in Cabin and 177 in Age and 2 in embarked
# print(train_data.isnull().mean().sort_values(ascending=False)) # percentage wise null value, we can see 77% null value in cabin attributes
# sns.countplot(train_data["Survived"])# Menas approx more than 500 died and approx 350 survived, hmmm means maximum died
# sns.countplot(x="Survived",hue='Sex',data=train_data) # survival of women are much higher
train_data['Sex']=pd.get_dummies(train_data['Sex'])# Dummy vairable add am many dummy number to catogerical value as many as different catogery
#i our case we had male and female as two different catogery so we have used dummy instead of dummy we can
# also use replace method of python as train_data['Sex'].replace('male',0,inplace=True)
# and train_data['Sex'].replace('female',1,inplace=True)
# sns.heatmap(train_data.isnull(), yticklabels = False, cbar = False, cmap = 'plasma')# heatmap of null value data via heatmap
# sns.countplot(train_data['Pclass'])# to check distribution of class
# print(train_data.values)
# sns.heatmap(train_data.corr(), annot = True)# to check co relation
# print(train_data.groupby('Pclass')['Age'].median())
# print(train_data['Name'].values)
train_data.loc[train_data['Age'].isnull(),'Age']=train_data.groupby("Pclass")['Age'].transform('median')
#The loc() function is used to access a group of rows and columns by label(s) or a boolean array.
# print(train_data['Age'].isnull().sum())# now no null values
# sns.distplot(train_data['Age'])
plt.show()
train_data.drop('Cabin',axis=1,inplace=True)
# from sklearn import impute
from statistics import mode
train_data['Embarked']=train_data['Embarked'].fillna(mode(train_data["Embarked"])) # imputing mode of categorical data as we can not apply
# mean and medain as this is catogerical data.
# train_data.info()
# print(train_data.isnull().sum())
# print(train_data["Embarked"])
# train_data["Embarked"]=pd.get_dummies(train_data["Embarked"])# this ia also a way but this is not alz preferd.
import warnings
warnings.filterwarnings('ignore')
# print(train_data["Embarked"].unique())# S,C,Q
train_data["Embarked"][train_data["Embarked"]=="S"]=0
# print(train_data[train_data["Embarked"]!="S"])# select * from train_data where embarked != "S"
train_data["Embarked"][train_data["Embarked"]=="C"]=1
train_data["Embarked"][train_data["Embarked"]=="Q"]=2
# print(train_data["Embarked"])
test_data=pd.DataFrame(input_file2)
train_data.drop('Name',axis=1,inplace=True)
train_data.drop('Ticket',axis=1,inplace=True)
train_data_target=train_data['Survived']
train_data.drop('Survived',axis=1,inplace=True)
from sklearn.model_selection import train_test_split
trainx,testx,trainy,testy=train_test_split(train_data,train_data_target,train_size=0.8,random_state=32)
# print(testy)
from sklearn.linear_model import LinearRegression
log_reg=LinearRegression()
log_reg.fit(trainx,trainy)
predictedq=log_reg.predict(testx)
from sklearn.metrics import mean_squared_error
acc=mean_squared_error(testy,predictedq)
# print(acc)
lin_rmse=np.sqrt(acc)
# print(lin_rmse)



from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression(max_iter=1000)
log_reg.fit(trainx,trainy)
prediction=log_reg.predict(testx)
# print(round(np.mean(prediction),2))
from sklearn.metrics import confusion_matrix
# print(confusion_matrix(testy,prediction))
from sklearn.metrics import accuracy_score
# print('manually calculated accuracy score',((90+49)/(90+49+22+18))) # 77.65 percent accuracy
# print(accuracy_score(testy,prediction))

#https://www.kaggle.com/kernelgenerator/titanic-tutorial-for-beginners-part-3
# but before doing that we can do feature engineering
# there we can add both feature sibligs and parent to total number of family onboarded.
# we can also change name and cabin to categorical value. but there also we can see total distinct cabnin and if any particualr cabin is repeating
# than we can replace other cabin name with other and then we will have
# 4 or 5 different cabin sets(like in train- AC 1 tear,AC 2 Tear, AC 3 tear and other)


"""Magic Weapon #2: Cross-Validation
Cross-Validation is essentially a more advanced form of our fundamental local validation method above.
 It protects against overfitting, that's why it is often the key to win a Kaggle competition. In a nutshell, 
 it is a resampling method which tells us how well our model would generalize to unseen data. This is achieved by fixing a number 
of partitions of the dataset called folds, predicting each fold separately, and averaging the predictions in the end."""

from sklearn.model_selection import KFold
kf=KFold(n_splits=5,random_state=32)
from sklearn.model_selection import cross_val_score
score=(cross_val_score(log_reg,testx,testy,cv=kf).mean())
# print(score) #79.3%

#lets do hyperparameter tunning Magic number 3
from sklearn.ensemble import RandomForestClassifier
ran_fst=RandomForestClassifier(random_state=32)
param_grid = {
    'criterion' : ['gini', 'entropy'],
    'n_estimators': [100, 300, 500],
    'max_features': ['auto', 'log2'],
    'max_depth' : [3, 5, 7]
}
"""'criterion' : A function which measures the quality of a split.
'n_estimators': The number of trees of our random forest.
'max_features': The number of features to choose when looking for the best way of splitting.
'max_depth' : the maximum depth of a decision tree."""

from sklearn.model_selection import GridSearchCV

ran_fst_cv=GridSearchCV(estimator=ran_fst,param_grid=param_grid,cv=5)
ran_fst_cv.fit(trainx,trainy)
# print('best parameters',ran_fst_cv.best_params_) # best parameters {'criterion': 'gini', 'max_depth': 3,
# 'max_features': 'log2', 'n_estimators': 300}
# print('best estimator',ran_fst_cv.best_estimator_)
# print('best index',ran_fst_cv.best_index_)
# print('best score',ran_fst_cv.best_score_)
ran_fst_final_mod=RandomForestClassifier(random_state=2,criterion='gini',max_depth=3,max_features='log2',n_estimators=300)
ran_fst_final_mod.fit(trainx,trainy)
final_prediction=ran_fst_final_mod.predict(testx)
print(accuracy_score(testy,final_prediction))