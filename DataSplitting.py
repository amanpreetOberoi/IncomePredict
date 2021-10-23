from sklearn.model_selection import train_test_split
from application_logging import logger


class DataSplitting:

    def __init__(self):
        self.logging =logger.App_Logger()
        self.fileread=open('TrainingLogs\ReadDataLogs.txt','a+')

    def split_data(self,X,y):
        try:
            self.logging.log(self.fileread,"Seperating independent and dependent features statrted!!")
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100)
            self.logging.log(self.fileread, "Data Splitting completed!!")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logging.log(self.fileread, "Error occur while splitting data!!")









