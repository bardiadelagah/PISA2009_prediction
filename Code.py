import pandas as pd
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from myClass import myClass

myInstance = myClass()

tr = pd.read_csv("pisa2009train.csv")
test = pd.read_csv("pisa2009test.csv")

tempTr = tr
tempTest = test

trInd_with_nan = tempTr.index[tempTr.isnull().any(axis=1)]
tempTr.drop(trInd_with_nan, 0, inplace=True)

trInd_with_nan = tempTest.index[tempTest.isnull().any(axis=1)]
tempTest.drop(trInd_with_nan, 0, inplace=True)
'''
mylist1 = tempTr.dropna()
tempTr = mylist1

mylist2 = tempTest.dropna()
tempTest = mylist1
'''

mylist1 = tempTr.replace({'raceeth': {'nan': 0,
                                      'White': 1,
                                      'Black': 2,
                                      'Hispanic': 3,
                                      'Asian': 4,
                                      'More than one race': 5,
                                      'American Indian/Alaska Native': 6,
                                      'Native Hawaiian/Other Pacific Islander': 7,
                                      }})

mylist2 = tempTest.replace({'raceeth': {'nan': 0,
                                        'White': 1,
                                        'Black': 2,
                                        'Hispanic': 3,
                                        'Asian': 4,
                                        'More than one race': 5,
                                        'American Indian/Alaska Native': 6,
                                        'Native Hawaiian/Other Pacific Islander': 7,
                                        }})
tempTr = mylist1
tempTest = mylist2
print(tempTr.columns)
print("shape of data: " + str(tempTr.shape))  # print(len(tempTr.index))

[min_maxNormalization, mean_Normalization] = myInstance.normalization(tempTr)
# tempTr = pd.DataFrame(data=min_maxNormalization, columns=tempTr.columns)

[min_maxNormalization, mean_Normalization] = myInstance.normalization(tempTest)
# tempTest = pd.DataFrame(data=min_maxNormalization, columns=tempTest.columns)

trV = tempTr.values
testV = tempTest.values

x_train = trV[:, :22]
y_train = trV[:, 23]
x_test = testV[:, :22]
y_test = testV[:, 23]

sgd = lm.SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='adaptive')

param_range = [0.01, 0.02, 0.03, 0.04]
param_name = 'eta0'
tuned_parameters = {param_name: param_range}

clf = GridSearchCV(estimator=sgd, param_grid=tuned_parameters, cv=5)

clf.fit(x_train, y_train)
# clf.predict(x_test, y_test)

train_error = clf.score(x_train, y_train)
print("train error: " + str(train_error))
test_error = clf.score(x_test, y_test)
print("test error: " + str(test_error))

print("best estimator : " + str(clf.best_estimator_))

best_sgd = clf.best_estimator_
title = 'for estimator by eta0 = ' + str(clf.best_params_['eta0'])
myInstance.plot_learning_curve(best_sgd, title, x_train, y_train, cv=5)
plt.show()

score = np.array([])
eta0 = 0.0
for i in range(0, 4):
    print(i)
    eta0 = eta0 + 1
    sgd = lm.SGDRegressor(max_iter=1000, tol=1e-3, eta0=eta0, learning_rate='adaptive')
    sgd.fit(x_train, y_train)
    print("for eta0 = " + str(eta0))
    print(sgd.score(x_train, y_train))
    score = np.append(score, sgd.score(x_train, y_train))

x = clf.cv_results_['params']
mylist = [x[i]['eta0'] for i in range(0, len(x))]
mean_test_score = clf.cv_results_['mean_test_score']  # numpy.ndarray
print("mean test score : " + str(mean_test_score))
myInstance.showValidationAndLossScore(mylist, score, mean_test_score)

plt.hist(tempTr['schoolSize'])
plt.xlabel('schoolSize')
plt.ylabel('frequency')
plt.show()
plt.hist(tempTr['minutesPerWeekEnglish'])
plt.xlabel('minutesPerWeekEnglish')
plt.ylabel('frequency')
plt.show()


var = statistics.variance(tempTr['schoolSize'])
mean = sum(tempTr['schoolSize']) / len(tempTr['schoolSize'])
print(mean)
print(var)

var = statistics.variance(tempTr['minutesPerWeekEnglish'])
mean = sum(tempTr['minutesPerWeekEnglish']) / len(tempTr['minutesPerWeekEnglish'])
print(mean)
print(var)
