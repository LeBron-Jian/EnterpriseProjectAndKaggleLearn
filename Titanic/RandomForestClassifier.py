import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
 
warnings.filterwarnings('ignore')
 
 
def load_dataset(trainfile, testfile):
    train = pd.read_csv(trainfile)
    test = pd.read_csv(testfile)
    train['Age'] = train['Age'].fillna(train['Age'].median())
    test['Age'] = test['Age'].fillna(test['Age'].median())
    # replace all the occurences of male with the number 0
    train.loc[train['Sex'] == 'male', 'Sex'] = 0
    train.loc[train['Sex'] == 'female', 'Sex'] = 1
    test.loc[test['Sex'] == 'male', 'Sex'] = 0
    test.loc[test['Sex'] == 'female', 'Sex'] = 1
    # .fillna() 为数据填充函数  用括号里面的东西填充
    train['Embarked'] = train['Embarked'].fillna('S')
    train.loc[train['Embarked'] == 'S', 'Embarked'] = 0
    train.loc[train['Embarked'] == 'C', 'Embarked'] = 1
    train.loc[train['Embarked'] == 'Q', 'Embarked'] = 2
    test['Embarked'] = test['Embarked'].fillna('S')
    test.loc[test['Embarked'] == 'S', 'Embarked'] = 0
    test.loc[test['Embarked'] == 'C', 'Embarked'] = 1
    test.loc[test['Embarked'] == 'Q', 'Embarked'] = 2
    test['Fare'] = test['Fare'].fillna(test['Fare'].median())
    traindata, trainlabel = train.drop('Survived', axis=1), train['Survived']  # train.pop('Survived')
    testdata = test
    print(traindata.shape, trainlabel.shape, testdata.shape)
    # (891, 11) (891,) (418, 11)
    return traindata, trainlabel, testdata
 
 
def random_forestclassifier_train(traindata, trainlabel, testdata):
    predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    traindata, testdata = traindata[predictors], testdata[predictors]
    # print(traindata.shape, trainlabel.shape, testdata.shape)  # (891, 7) (891,) (418, 7)
    # print(testdata.info())
    trainSet, testSet, trainlabel, testlabel = train_test_split(traindata, trainlabel,
                                                                test_size=0.2, random_state=12345)
    # initialize our algorithm class
    clf = RandomForestClassifier(random_state=1, n_estimators=100,
                                 min_samples_split=4, min_samples_leaf=2)
    # training the algorithm using the predictors and target
    clf.fit(trainSet, trainlabel)
    test_accuracy = clf.score(testSet, testlabel) * 100
    print("正确率为   %s%%" % test_accuracy)  # 正确率为   81.56424581005587%
 
 
if __name__ == '__main__':
    trainfile = 'data/titanic_train.csv'
    testfile = 'data/test.csv'
    traindata, trainlabel, testdata = load_dataset(trainfile, testfile)
    random_forestclassifier_train(traindata, trainlabel, testdata)