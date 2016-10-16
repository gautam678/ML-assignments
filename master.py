from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import csv
import pandas as dp

def load():
    data=[]
    with open("salary_labeled.csv", 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            data.append(row)
        new_data=dp.DataFrame(data)
        cate_data=new_data[[1,3,5,6,7,8,9,13]]
        num_data=new_data[[0,2,4,10,11,12]]
        vec_x_cat_train = dp.get_dummies(cate_data)
        print vec_x_cat_train.shape





load()
