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
 
def linear_regression_test(traindata, trainlabel, testdata):
    traindata = pd.read_csv(trainfile)
    test = pd.read_csv(testfile)
    traindata['Age'] = traindata['Age'].fillna(traindata['Age'].median())
    test['Age'] = test['Age'].fillna(test['Age'].median())
    # replace all the occurences of male with the number 0
    traindata.loc[traindata['Sex'] == 'male', 'Sex'] = 0
    traindata.loc[traindata['Sex'] == 'female', 'Sex'] = 1
    # .fillna() 为数据填充函数  用括号里面的东西填充
    traindata['Embarked'] = traindata['Embarked'].fillna('S')
    traindata.loc[traindata['Embarked'] == 'S', 'Embarked'] = 0
    traindata.loc[traindata['Embarked'] == 'C', 'Embarked'] = 1
    traindata.loc[traindata['Embarked'] == 'Q', 'Embarked'] = 2
    # the columns we'll use to predict the target
    all_variables = ['PassengerID', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
                     'Ticket', 'Fare', 'Cabin', 'Embarked']
    predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    # traindata, testdata = traindata[predictors], testdata[predictors]
    alg = LinearRegression()
    kf = KFold(n_splits=3, random_state=1)
    predictions = []
    for train_index, test_index in kf.split(traindata):
        # print(train_index, test_index)
        train_predictors = (traindata[predictors].iloc[train_index, :])
        train_target = traindata['Survived'].iloc[train_index]
        alg.fit(train_predictors, train_target)
        test_predictions = alg.predict(traindata[predictors].iloc[test_index, :])
        predictions.append(test_predictions)
    # print(type(predictions))
    predictions = np.concatenate(predictions, axis=0)  #<class 'numpy.ndarray'>
    # print(type(predictions))
    predictions[predictions > 0.5] = 1
    predictions[predictions < 0.5] = 1
    accuracy = sum(predictions[predictions == traindata['Survived']]/len(predictions))
    print(accuracy)  # 0.3838383838383825
 
def linear_regression_train(traindata, trainlabel, testdata):
    # the columns we'll use to predict the target
    all_variables = ['PassengerID', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
                     'Ticket', 'Fare', 'Cabin', 'Embarked']
    predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    traindata, testdata = traindata[predictors], testdata[predictors]
    print(traindata.shape, trainlabel.shape, testdata.shape)  # (891, 7) (891,) (418, 7)
    print(testdata.info())
    trainSet, testSet, trainlabel, testlabel = train_test_split(traindata, trainlabel,
                                                                test_size=0.2, random_state=12345)
    # initialize our algorithm class
    clf = LinearRegression()
    # training the algorithm using the predictors and target
    clf.fit(trainSet, trainlabel)
    test_accuracy = clf.score(testSet, testlabel) * 100
    print("正确率为   %s%%" % test_accuracy)  # 正确率为   37.63003367180264%
    # res = clf.predict(testdata)
 
 
 
if __name__ == '__main__':
    trainfile = 'data/titanic_train.csv'
    testfile = 'data/test.csv'
    traindata, trainlabel, testdata = load_dataset(trainfile, testfile)
    # print(traindata.shape[1])  # 11
    linear_regression_train(traindata, trainlabel, testdata)
    # linear_regression_test(traindata, trainlabel, testdata)