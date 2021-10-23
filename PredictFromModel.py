from application_logging.logger import App_Logger
from ModelFinder import finder
import pandas as pd
class prediction:

    def __init__(self):
        self.file_read = open("ModelLogs/Model_Predict.txt", 'a+')
        self.log_writer = App_Logger()

    def predict_results(self,data):
        try:
            predictions="Income is "
            self.log_writer.log(self.file_read, 'Start of Prediction')
            model_load=finder.ModelFinder()
            model = model_load.load_model()
            to_predict = np.array(data).reshape(1, 44)
            result=model.predict(to_predict)
            self.log_writer.log(self.file_read, result)
            self.log_writer.log(self.file_read, 'Model loaded successfully')
            if result == 0:
                predictions.concat('<=50K')
            else:
                predictions.concat('>50K')

            self.log_writer.log(self.file_read, 'End of Prediction')
            return predictions
        except Exception as ex:
            self.log_writer.log(self.file_read, 'Error occurred while running the prediction!! Error:: %s' % ex)


