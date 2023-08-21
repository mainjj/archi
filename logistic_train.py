# Logistic Regression train by RandomSearchCV
# X,y in (X_data.csv,y_data.csv) have to headless (without columns,index)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon
import pandas as pd
import pickle
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

def train(data_path,save_name,n_iter):
    random_state = 777

    #load csv data (headless)
    df = pd.read_csv(data_path,header=None)
    raw_y = df.loc[:,0]
    raw_X = df.loc[:,1:]
    
    #if Data shape is not equal raise error
    if len(raw_X) != len(raw_y):
        raise Exception(f"data is not equal\nX: {len(raw_X)}\ty: {len(raw_y)}")
    #split train,test data
    X_train , X_test, y_train , y_test = train_test_split(raw_X, raw_y, test_size=0.2, random_state=random_state)
    print(f"Successful load data\nX data: {len(X_train)+len(X_test)}\ty data: {len(y_train)+len(y_test)}")
    
    #LogisticRegression
    lr_clf = LogisticRegression(random_state=random_state)

    #set params for randomsearch 
    params={'solver':['liblinear','lbfgs','saga','newton-cg','sag'],
            'penalty':['l2', 'l1','elasticnet'],
            'C':expon(scale=1.0),
            'max_iter':[1000]}
    print(f"start RandomSearchCV\nparams: {params}\n")
    #RandomSearch
    
    rand_clf = RandomizedSearchCV(
        lr_clf,
        param_distributions=params,
        n_iter = n_iter,
        cv = 5,        
        scoring='accuracy',
        random_state = 2,
        verbose=1,
        refit=True
    )
    rand_clf.fit(X_train, y_train)
    #print result
    pred = rand_clf.predict(X_test)
    print("RandomSearchCV RESULT: {0}\nBest Acc: {1:.3f}".format(rand_clf.best_estimator_,rand_clf.best_score_))
    print("-"*20)
    print("Test ACC : {0:.3f}".format(accuracy_score(y_test,pred)))
    print('accuracy = {0:.3f}'.format(accuracy_score(y_test, pred)))
    print('precision = {0:.3f}'.format((precision_score(y_test, pred))))
    print('recall = {0:.3f}'.format(recall_score(y_test, pred)))
    print('f1 score = {0:.3f}'.format(f1_score(y_test, pred)))
    print('confusion matrix = \n', confusion_matrix(y_test, pred))
    print("-"*20)
    #save model.pickle
    pickle.dump(rand_clf, open(save_name, "wb"))
    print(f"\nsave model in {save_name}\n")



if __name__=="__main__":    
    #############################################################################################
    parser = argparse.ArgumentParser(description='Logistic Regression train by RandomSearchCV\nX,y data.csv have to headless(without columns,index)')
    parser.add_argument('-d','--data', help='data.csv path')
    parser.add_argument('-i','--iter', type=int, default=500, help='RandomSearch iter | default=500')
    parser.add_argument('-p','--project', default='./',help='save dir path | default = ./')
    parser.add_argument('-n','--name', help='save model name | it will be saved like /project/name.pickle')
    args = parser.parse_args()

    data_path,n_iter,save_dir,name = args.data,args.iter,args.project,args.name
    #############################################################################################
    train(data_path,os.path.join(save_dir,f'{name}.pickle'), n_iter)











