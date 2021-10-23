import pandas as pd
from application_logging.logger import App_Logger

class Data_Getter_Pred:
    def __init__(self,path):
        self.prediction_file='Prediction_File_Path\Adult.csv'
        self.fileread = open('PredictionLogs\PredictionGetDataLogs.txt', 'a+')
        self.logWriter = App_Logger()

    def get_data(self):
        
        self.logWriter.log(self.fileread,'Entered the get_data method of the Data_Getter_Pred class')
        try:
            self.data= pd.read_csv(self.prediction_file) # reading the data file
            print(self.data)
            self.logWriter.log(self.fileread,'Data Load Successful.Exited the get_data method of the Data_Getter_Pred class')
            return self.data
        except Exception as e:
            self.logWriter.log(self.fileread,'Exception occured in get_data method of the Data_Getter class. Exception message: '+str(e))
            self.logWriter.log(self.fileread,
                                   'Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')




