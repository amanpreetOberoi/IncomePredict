from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics  import roc_auc_score,accuracy_score
from application_logging import logger
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
import os
import pickle
import shutil

class ModelFinder:

    def __init__(self):
        self.log_writer=logger.App_Logger()
        self.file_read=open("ModelLogs\model_logs.txt","a+")
        self.log_reg=LogisticRegression()
        self.xgb=XGBClassifier()
        self.rf=RandomForestClassifier()
        self.gnb=GaussianNB()
        self.model_directory="models/"

    def save_model(self,filename,model):
        self.log_writer.log(self.file_read, 'Entered the save_model method of the File_Operation class')
        try:
            path = os.path.join(self.model_directory, filename)
            if os.path.isdir(path):  # remove previously existing models
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)  #
            with open(path + '/' + filename + '.sav',
                      'wb') as f:
                pickle.dump(model, f)  # save the model to file
            self.log_writer.log(self.file_read,
                                   'Model File ' + filename + ' saved. Exited the save_model method of the Model_Finder class')

            return 'success'
        except Exception as e:
            self.log_writer.log(self.file_read,
                                   'Exception occured in save_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))

    def load_model(self):
        self.log_writer.log(self.file_read, 'Entered the load_model method of the File_Operation class')
        try:
            with open(self.model_directory + 'XGBoost' + '/' + 'XGBoost' + '.sav','rb') as f:
                self.log_writer.log(self.file_read,
                                       'Model File XGBoost loaded. Exited the load_model method of the Model_Finder class')
                return pickle.load(f)
        except Exception as e:
            self.log_writer.log(self.file_read,'Exception occured in load_model method of the Model_Finder class. Exception message:  ' + str(e))

    def best_param_for_logistic_regression(self,x_train,y_train):
        try:
            self.log_writer.log(self.file_read, "Entered in best params for logistic regression function !!")
            solver = ['liblinear']
            penalty = ['l1','l2']
            c_values = [1.0, 0.1, 0.01]
            # define grid search
            self.grid = dict( penalty=penalty, C=c_values,solver=solver)
            #self.cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
            self.grid_search = RandomizedSearchCV(estimator=self.log_reg,param_distributions=self.grid ,scoring='accuracy', random_state=1)
            self.grid_search.fit(x_train,y_train)
            best_params=self.grid_search.best_params_
            print(best_params)
            solvers=self.grid_search.best_params_['solver']
            print(solvers)
            penalty=self.grid_search.best_params_['penalty']
            print(penalty)
            c_values=self.grid_search.best_params_['C']
            print(c_values)
            self.log_reg=LogisticRegression(solver=solvers,penalty=penalty,C=c_values)
            self.log_reg.fit(x_train,y_train)
            print(self.log_reg)
            self.log_writer.log(self.file_read, "Exiting best params for logistic regression function !!")
            return self.log_reg
        except Exception as e:
            self.log_writer.log(self.file_read, "Error occurred while executing best param for logisitc regression{}".format(e))

    def get_best_param_for_RandomForest(self,x_train,y_train):
        try:
            self.log_writer.log(self.file_read, "Entered in best params for random forest function !!")
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]
            # Create the random grid
            self.random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}
            #self.cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
            print(self.random_grid)
            self.grid_search = RandomizedSearchCV(estimator=self.rf,param_distributions=self.random_grid ,scoring='accuracy', random_state=1,n_iter=5,n_jobs=-1)
            self.grid_search.fit(x_train,y_train)
            best_params=self.grid_search.best_params_
            n_estimators=self.grid_search.best_params_['n_estimators']
            max_features = self.grid_search.best_params_['max_features']
            max_depth = self.grid_search.best_params_['max_depth']
            min_samples_split = self.grid_search.best_params_['min_samples_split']
            min_samples_leaf=self.grid_search.best_params_['min_samples_leaf']
            bootstrap=self.grid_search.best_params_['bootstrap']

            rf_clf=RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, bootstrap=bootstrap)
            rf_clf.fit(x_train, y_train)
            self.log_writer.log(self.file_read, "Exiting best params for random forest classifier function !!")
            return rf_clf
        except Exception as e:
            self.log_writer.log(self.file_read, "Error occurred while executing best param for random forest{}".format(e))

    def get_best_param_for_XGBoost(self,x_train,y_train):
        try:
            self.log_writer.log(self.file_read, "Entered in best params for xg boost function !!")
            self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                          "max_depth": range(2, 10, 1)}
            self.grid_search = RandomizedSearchCV(estimator=self.xgb,param_distributions=self.param_grid ,cv=5,scoring='accuracy', random_state=1,n_iter=5,n_jobs=-1,)
            self.grid_search.fit(x_train,y_train)
            best_params=self.grid_search.best_params_
            n_estimators=self.grid_search.best_params_['n_estimators']
            criterion = self.grid_search.best_params_['criterion']
            max_depth = self.grid_search.best_params_['max_depth']
            xgb_clf=XGBClassifier(n_estimators=n_estimators,criterion=criterion,max_depth=max_depth)
            xgb_clf.fit(x_train,y_train)
            self.log_writer.log(self.file_read, "Exiting best params for xgboost classifier function !!")
            return xgb_clf
        except Exception as e:
            self.log_writer.log(self.file_read, "Error occurred while executing best param for xg boost{}".format(e))

    def get_best_params_for_naive_bayes(self,x_train,y_train):
        try:
            self.log_writer.log(self.file_read,
                                'Entered the get_best_params_for_naive_bayes method of the Model_Finder class')
            # initializing with different combination of parameters
            self.param_grid = {"var_smoothing": [1e-9,0.1, 0.001, 0.5,0.05,0.01,1e-8,1e-7,1e-6,1e-10,1e-11]}

            #Creating an object of the Grid Search class
            self.grid = RandomizedSearchCV(estimator=self.gnb, param_distributions=self.param_grid, verbose=3)
            #finding the best parameters
            self.grid.fit(x_train,y_train)

            #extracting the best parameters
            self.var_smoothing = self.grid.best_params_['var_smoothing']


            #creating a new model with the best parameters
            self.gnb = GaussianNB(var_smoothing=self.var_smoothing)
            # training the mew model
            self.gnb.fit(x_train,y_train)
            self.log_writer.log(self.file_read,
                                   'Naive Bayes best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')

            return self.gnb
        except Exception as e:
            self.log_writer.log(self.file_read,'Exception occured in get_best_params_for_naive_bayes method of the Model_Finder class. Exception message:  ' + str(e))

    def get_best_model(self,x_train,x_test,y_train,y_test):
        self.log_reg=self.best_param_for_logistic_regression(x_train, y_train)
        y_pred=self.log_reg.predict(x_test)
        accuracy_lr=accuracy_score(y_test,y_pred)
        print("logistic regression accuracy :", accuracy_lr)

        self.xgb=self.get_best_param_for_XGBoost(x_train,y_train)
        y_pred=self.xgb.predict(x_test)
        accuracy_xgb=accuracy_score(y_test,y_pred)
        print("XG Boost Accuracy",accuracy_xgb)

        self.rf = self.get_best_param_for_RandomForest(x_train, y_train)
        y_pred = self.rf.predict(x_test)
        accuracy_rf = accuracy_score(y_test, y_pred)
        print("Random Forest Accuracy",accuracy_rf)

        self.gnb=self.get_best_params_for_naive_bayes(x_train, y_train)
        y_pred = self.gnb.predict(x_test)
        accuracy_gnb = accuracy_score(y_test, y_pred)
        print("Gaussian Naive Bayes",accuracy_gnb)

        if(accuracy_lr> accuracy_rf and accuracy_lr >accuracy_gnb and accuracy_lr > accuracy_xgb):
            print('logistic regression')
            self.save_model('LogisticRegression',self.log_reg)
        else:
            if (accuracy_rf > accuracy_lr and accuracy_rf > accuracy_gnb and accuracy_rf > accuracy_xgb):
                print('Random Forest')
                self.save_model('RandomForest',self.rf)
            else:
                if (accuracy_xgb > accuracy_lr and accuracy_xgb > accuracy_gnb and accuracy_xgb > accuracy_rf):
                    print('XG Boost')
                    self.save_model('XGBoost', self.xgb)
                else:
                    if (accuracy_gnb > accuracy_lr and accuracy_gnb > accuracy_rf and accuracy_gnb > accuracy_xgb):
                        print('Gaussian Naive_Bayes')
                        print('XG Boost')
                        self.save_model('GaussianNB', self.gnb)