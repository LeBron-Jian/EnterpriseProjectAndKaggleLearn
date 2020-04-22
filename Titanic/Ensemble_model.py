import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
import re
 
warnings.filterwarnings('ignore')
 
 
# A function to get the title from a name
def get_title(name):
    # use a regular expression to search for a title
    title_search = re.search('([A-Za-z]+)\.', name)
    # if the title exists, extract and return it
    if title_search:
        return title_search.group(1)
    return ''
 
 
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
    # generating a familysize column  是指所有的家庭成员
    train['FamilySize'] = train['SibSp'] + train['Parch']
    test['FamilySize'] = test['SibSp'] + test['Parch']
 
    # the .apply method generates a new series
    train['NameLength'] = train['Name'].apply(lambda x: len(x))
    test['NameLength'] = test['Name'].apply(lambda x: len(x))
 
    titles_train = train['Name'].apply(get_title)
    titles_test = test['Name'].apply(get_title)
    title_mapping = {
        'Mr': 1,
        'Miss': 2,
        'Mrs': 3,
        'Master': 4,
        'Rev': 6,
        'Dr': 5,
        'Col': 7,
        'Mlle': 8,
        'Ms': 2,
        'Major': 7,
        'Don': 9,
        'Countess': 10,
        'Mme': 8,
        'Jonkheer': 10,
        'Sir': 9,
        'Dona': 9,
        'Capt': 7,
        'Lady': 10,
    }
    for k, v in title_mapping.items():
        titles_train[titles_train == k] = v
    train['Title'] = titles_train
    for k, v in title_mapping.items():
        titles_test[titles_test == k] = v
    test['Title'] = titles_test
    # print(pd.value_counts(titles_train))
    traindata, trainlabel = train.drop('Survived', axis=1), train['Survived']  # train.pop('Survived')
    testdata = test
    print(traindata.shape, trainlabel.shape, testdata.shape)
    # (891, 11) (891,) (418, 11)
    return traindata, trainlabel, testdata
 
 
 
def emsemble_model_train(traindata, trainlabel, testdata):
    # the algorithms we want to ensemble
    # we're using the more linear predictors for the logistic regression,
    # and everything with the gradient boosting classifier
    algorithms = [
        [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
         ['Pclass', 'Sex', 'Fare', 'FamilySize', 'Title', 'Age', 'Embarked', ]],
        [LogisticRegression(random_state=1),
         ['Pclass', 'Sex', 'Fare', 'FamilySize', 'Title', 'Age', 'Embarked', ]]
    ]
    # initialize the cross validation folds
    kf = KFold(n_splits=3, random_state=1)
    predictions = []
    for train_index, test_index in kf.split(traindata):
        # print(train_index, test_index)
        full_test_predictions = []
        for alg, predictors in algorithms:
            train_predictors = (traindata[predictors].iloc[train_index, :])
            train_target = trainlabel.iloc[train_index]
            alg.fit(train_predictors, train_target)
            test_predictions = alg.predict(traindata[predictors].iloc[test_index, :])
            full_test_predictions.append(test_predictions)
        # use a simple ensembling scheme
        test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
        test_predictions[test_predictions <= 0.5] = 0
        test_predictions[test_predictions >= 0.5] = 1
        predictions.append(test_predictions)
    predictions = np.concatenate(predictions, axis=0)
    # compute accuracy bu comparing to the training data
    accuracy = sum(predictions[predictions == trainlabel]) / len(predictions)
    print(accuracy)
 
def emsemble_model_train(traindata, trainlabel, testdata):
    # the algorithms we want to ensemble
    # we're using the more linear predictors for the logistic regression,
    # and everything with the gradient boosting classifier
    predictors = ['Pclass', 'Sex', 'Fare', 'FamilySize', 'Title', 'Age', 'Embarked', ]
    traindata, testdata = traindata[predictors], testdata[predictors]
    trainSet, testSet, trainlabel, testlabel = train_test_split(traindata, trainlabel,
                                                                test_size=0.2, random_state=12345)
    # initialize our algorithm class
    clf1 = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
    clf2 = LogisticRegression(random_state=1)
    # training the algorithm using the predictors and target
    clf1.fit(trainSet, trainlabel)
    clf2.fit(trainSet, trainlabel)
    test_accuracy1 = clf1.score(testSet, testlabel) * 100
    test_accuracy2 = clf2.score(testSet, testlabel) * 100
    print(test_accuracy1, test_accuracy2)  # 78.77094972067039   80.44692737430168
    print("正确率为   %s%%" % ((test_accuracy1+test_accuracy2)/2))  # 正确率为   79.60893854748603%
 
 
if __name__ == '__main__':
    trainfile = 'data/titanic_train.csv'
    testfile = 'data/test.csv'
    traindata, trainlabel, testdata = load_dataset(trainfile, testfile)
    emsemble_model_train(traindata, trainlabel, testdata)