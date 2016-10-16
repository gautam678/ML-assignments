# Starting code for CS6316/4501 HW3, Fall 2016
# By Weilin Xu

import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
import random
import csv
import pandas as pd
import os


# Attention: You're not allowed to use the model_selection module in sklearn.
#            You're expected to implement it with your own code.
# from sklearn.model_selection import GridSearchCV

class SvmIncomeClassifier:
    def __init__(self):
        random.seed(0)

    def load_data(self, csv_fpath):
        col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                     'hours-per-week', 'native-country','class']
        col_names_y = ['label']

        numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                          'hours-per-week']
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                            'race', 'sex', 'native-country']
        data=[]
        # Remove the following lines and write your own code for imputing
        new_data = pd.read_csv(csv_fpath,delimiter=',',header=None, names=col_names_x, na_values=[" ?"])
        imp = preprocessing.Imputer(missing_values="NaN", strategy='most_frequent', axis=0)

        # Splitting categorical data and Numerical data
        cate_data=new_data[[1,3,5,6,7,8,9,13]]
        num_data=new_data[[0,2,4,10,11,12]]
        #MARK THIS POINT FOR WORKING ON LATER (scaling)



        # Adding a y value to store results
        y=new_data[[14]]
        # Creating a sparse matrix using get_dummies()
        vec_x_cat_train = pd.get_dummies(cate_data)
        # Concating the categorical data and the numerical data into a single x matrix
        x = pd.concat([vec_x_cat_train, num_data], axis=1)
        imp.fit(x)
        pass
        return x,y   

    def train_and_select_model(self, training_csv):
        x_train, y_train = self.load_data(training_csv)
        # Load test data and fill in categorical data in the training set too
        xtemp,ytemp=self.load_data("salary.2Predict.csv")
        training_header=list(x_train.columns.values)
        count=0
        not_in=[]
        ind=[]
        for i in xtemp.columns.values:
            if i not in training_header:
                not_in.append(i)
                ind.append(count)
            count=count+1
        for i in range(len(ind)):
            x_train.insert(ind[i], not_in[i], 0)
        # Set seeds to retain values

        random.seed(37)
        # The code written should lopp through the parameter and find the best fit
        param_set = [
                     {'kernel': 'rbf', 'C': 1, 'degree': 1}
                     #{'kernel': 'rbf', 'C': 1, 'degree': 3},
                     #{'kernel': 'rbf', 'C': 1, 'degree': 5},
                     #{'kernel': 'rbf', 'C': 1, 'degree': 7},
        ]
        clf = SVC()
        for j in param_set:
            # Implementing 3-fold validation method
            for i in range(0,3):
                msk = np.random.rand(len(x_train)) <= 0.66
                training_x = x_train[msk]
                testing_x = x_train[~msk]
                training_y= y_train[msk]
                testing_y = y_train[~msk]
                training_x.as_matrix()
                training_y.as_matrix()
                testing_x.as_matrix()
                testing_y.as_matrix()
                clf=SVC(j['C'],j['kernel'],j['degree'])
                model=clf.fit(training_x,training_y)
                if best_score<=clf.score(testing_x,testing_y):
                    best_score=clf.score(testing_x,testing_y)
                    best_model=model
        return best_model, best_score

    def predict(self, test_csv, trained_model):
        x_test, _ = self.load_data(test_csv)
        predictions = trained_model.predict(x_test)
        return predictions

    def output_results(self, predictions):
        # 3. Upload your Python code, the predictions.txt as well as a report to Collab.
        # Hint: Don't archive the files or change the file names for the automated grading.
        with open('predictions.txt', 'w') as f:
            for pred in predictions:
                if pred == 0:
                    f.write('<=50K\n')
                else:
                    f.write('>50K\n')

if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    clf = SvmIncomeClassifier()
    trained_model, cv_score = clf.train_and_select_model(training_csv)
    print "The best model was scored %.2f" % cv_score
    predictions = clf.predict(testing_csv, trained_model)
    clf.output_results(predictions)


