import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings('ignore')


def load_dataset(trainfile, testfile):
    train = pd.read_csv(trainfile)
    test = pd.read_csv(testfile)
    # print(test.info())
    all_data = pd.concat([train, test], ignore_index=True)

    titles = all_data['Name'].apply(get_title)
    # print(pd.value_counts(titles))
    # map each title to an integer some titles are very rare
    # and are compressed into the same codes as other titles
    title_mapping = {
        'Mr': 2,
        'Miss': 3,
        'Mrs': 4,
        'Master': 1,
        'Rev': 5,
        'Dr': 5,
        'Col': 5,
        'Mlle': 3,
        'Ms': 4,
        'Major': 6,
        'Don': 5,
        'Countess': 5,
        'Mme': 4,
        'Jonkheer': 1,
        'Sir': 5,
        'Dona': 5,
        'Capt': 6,
        'Lady': 5,
    }
    for k, v in title_mapping.items():
        titles[titles == k] = v
        # print(k, v)
    all_data['Title'] = titles

    grouped = all_data.groupby(['Title'])
    median = grouped.Age.median()
    for i in range(len(all_data['Age'])):
        if pd.isnull(all_data['Age'][i]):
            all_data['Age'][i] = median[all_data['Title'][i]]
    # print(all_data['Age'])

    # generating a familysize column  是指所有的家庭成员
    all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch']

    # the .apply method generates a new series
    all_data['NameLength'] = all_data['Name'].apply(lambda x: len(x))
    # print(all_data['NameLength'])

    all_data['Embarked'] = all_data['Embarked'].fillna('S')
    # 缺失数据比较多，船舱号（Cabin）缺失值填充为U，表示未知（Uknow）
    all_data['Cabin'] = all_data['Cabin'].fillna('U')
    all_data['Fare'] = all_data['Fare'].fillna(all_data.Fare.median())

    all_data.loc[all_data['Embarked'] == 'S', 'Embarked'] = 0
    all_data.loc[all_data['Embarked'] == 'C', 'Embarked'] = 1
    all_data.loc[all_data['Embarked'] == 'Q', 'Embarked'] = 2

    all_data.loc[all_data['Sex'] == 'male', 'Sex'] = 0
    all_data.loc[all_data['Sex'] == 'female', 'Sex'] = 1

    traindata, testdata = all_data[:891], all_data[891:]
    # print(traindata.shape, testdata.shape, all_data.shape)  # (891, 15) (418, 15) (1309, 15)
    traindata, trainlabel = traindata.drop('Survived', axis=1), traindata['Survived']  # train.pop('Survived')
    testdata = testdata
    corrDf = all_data.corr()
    '''
        查看各个特征与生成情况（Survived）的相关系数，
        ascending=False表示按降序排列
    '''
    res = corrDf['Survived'].sort_values(ascending=False)
    print(res)
    return traindata, trainlabel, testdata


def Ticket_Label1(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0


def Fam_label1(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0


# A function to get the title from a name
def get_title(name):
    # use a regular expression to search for a title
    title_search = re.search('([A-Za-z]+)\.', name)
    # if the title exists, extract and return it
    if title_search:
        return title_search.group(1)
    return ''


def random_forestclassifier_train(traindata, trainlabel, testdata):
    predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title', 'NameLength']

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
    print("正确率为   %s%%" % test_accuracy)  # 正确率为   83.79888268156425%


def feature_extract(traindata, trainlabel, testdata):
    # the columns we'll use to predict the target
    predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                  'FamilySize', 'Title', 'NameLength']
    # perform feature selection
    traindata, testdata = traindata[predictors], testdata[predictors]
    selector = SelectKBest(f_classif, k=5)
    selector.fit(traindata, trainlabel)

    # get thr raw p-values for each feature, and transform from p-value into scores
    scores = -np.log10(selector.pvalues_)

    # plot the scores, see how Pclass Title Sex, and Fare are the best?
    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    plt.show()

    # # pick only the four best features
    # predictors = ['Pclass', 'Sex', 'Fare', 'Title', 'NameLength']
    trainSet, testSet, trainlabel, testlabel = train_test_split(traindata, trainlabel,
                                                                test_size=0.2, random_state=12345)

    clf = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_leaf=4, min_samples_split=8)
    clf.fit(trainSet, trainlabel)
    test_accuracy = clf.score(testSet, testlabel) * 100
    print("正确率为   %s%%" % test_accuracy)  # 正确率为   80.44692737430168%


if __name__ == '__main__':
    trainfile = 'data/titanic_train.csv'
    testfile = 'data/test.csv'
    traindata, trainlabel, testdata = load_dataset(trainfile, testfile)
    random_forestclassifier_train(traindata, trainlabel, testdata)
    # feature_extract(traindata, trainlabel, testdata)
