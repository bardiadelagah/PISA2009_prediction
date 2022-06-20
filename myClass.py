import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


class myClass:

    def raceethToInt1(self, dataFrame):
        raceeth = np.array([])
        raceeth = np.append(raceeth, dataFrame)
        number = np.arange(len(raceeth))
        df = pd.DataFrame(columns=['raceeth', 'number'])
        df2 = pd.DataFrame({"raceeth": raceeth,
                            "number": number})
        df = df.append(df2, ignore_index=True)
        return df

    def numberSwitchToraceeth(self, argument):
        switcher = {
            0: 'nan',
            1: 'Black',
            2: 'Hispanic',
            3: 'Asian',
            4: 'More than one race',
            5: 'American Indian/Alaska Native',
            6: 'Native Hawaiian/Other Pacific Islander',
        }
        return switcher.get(argument, "Invalid month")

    def raceethSwitchToNumber(self, argument):
        switcher = {
            'nan': 0,
            'White': 1,
            'Black': 2,
            'Hispanic': 3,
            'Asian': 4,
            'More than one race': 5,
            'American Indian/Alaska Native': 6,
            'Native Hawaiian/Other Pacific Islander': 7,
        }
        return int(switcher.get(argument, "Invalid month"))

    def normalization(self, df):
        mean = df.mean(axis=0, skipna=True)
        max = df.max(axis=0, skipna=True)
        min = df.min(axis=0, skipna=True)
        # print(mean)
        # print(mean[0])
        # print(len(mean))
        # print(mean[len(mean) - 1])
        counter1 = np.array([])
        counter2 = np.array([])
        h1 = np.array([[]])
        h2 = np.array([[]])
        i = 0
        j = 0
        [x, y] = df.shape
        mydataframe = pd.DataFrame
        # for i in range(0,len(tempTr.index)):
        for (columnName, columnData) in df.iteritems():
            # print('Colunm Name : ', columnName)
            # print('Column Contents : ', columnData.values)
            j = 0
            # for index, row in df.iterrows():
            for index in range(0, len(df.index)):
                # print('Column Contents : ', columnData.values[index])
                # print('===========', len(columnData.values))
                # print(row)
                # print(index)
                a1 = (columnData.values[index] - min[i]) / (max[i] - min[i])
                counter1 = np.append(counter1, a1)
                a2 = (columnData.values[index] - mean[i]) / (max[i] - min[i])
                counter2 = np.append(counter2, a2)
                j = j + 1
                # print(index)
                # print(row)
            i = i + 1
            # print()
            # h1 = np.append(h1, counter1)
            # h2 = np.append(h2, counter2)
            # counter1 = np.array([])
            # counter2 = np.array([])
        min_maxNormalization = counter1.reshape(j, i)
        mean_Normalization = counter2.reshape(j, i)
        # print(i)
        # print(j)
        # print(counter1.reshape(i, j))
        # print(counter2.reshape(i, j))

        return min_maxNormalization, mean_Normalization

    def showLossCurvePlot(self, estimator):
        y_array = estimator.loss_curve_
        x_array = [0]
        for i in range(1, len(y_array)):
            x_array.append(i)
        plt.title("loss curve")
        plt.ylabel('loss')
        plt.xlabel('number of iterations(epochs)')
        plt.plot(x_array, y_array, label="loss score")
        plt.legend(loc="best")
        plt.show()

    def plot_learning_curve(self, estimator, title, x, y, axes=None, ylim=None, cv=None,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 8)):
        # np.linspace(.1, 1.0, 5) [0.1   0.325 0.55  0.775 1.   ] np.array([0.00001,1.0])

        if axes is None:
            _, axes = plt.subplots(1, 1, figsize=(20, 5))

        axes.set_title(title)
        if ylim is not None:
            axes.set_ylim(*ylim)
        axes.set_xlabel("Training examples")
        axes.set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, x, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True)

        train_scores = 1 - train_scores
        test_scores = 1 - test_scores
        # print('train_sizes')
        # print(train_sizes)
        # print('train_scores')
        # print(train_scores)
        # print('test_scores')
        # print(test_scores(axes=0))
        # print(test_scores(axes=1))
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes.grid()
        '''        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        '''

        axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes.legend(loc="best")

        '''
        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")
        '''
        return plt

    def showValidationAndLossScore(self, x, loss, means):
        loss = [float(i) for i in loss]
        means = [float(i) for i in means]
        means = [1 - i for i in means]
        plt.plot(x, loss, label="Training score curve", color="darkorange")
        plt.plot(x, means, label="test score curve", color="blue")
        plt.title("Validation and loss Curve with different hidden layer in mlp")
        plt.ylabel('Score')
        plt.xlabel('eta0')
        plt.legend(loc="best")
        plt.show()

