import pandas as pd
from application_logging import logger


class ReadData:
    def __init__(self):
        self.fileread=open('TrainingLogs\ReadDataLogs.txt','a+')
        self.logWriter=logger.App_Logger()

    def readData(self):
        try:
            self.logWriter.log(self.fileread,"Reading Data Started!!")
            data=pd.read_csv('adult.csv')
            self.logWriter.log(self.fileread, "Reading Data Completed!!")
            return data
        except  Exception as e:
            self.logWriter.log(self.fileread, "Error while reading data!! ")



