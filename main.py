import pandas as pd
import os
from ModelFinder import finder
from application_logging.logger import App_Logger
from application_logging import logger
from flask import Flask, request, render_template,Response
from flask_cors import cross_origin,CORS
import ReadData
from DataSplitting import DataSplitting
from Training_Data_Traonsfrmation import Preprocessing
from ModelFinder.finder import ModelFinder
from predictionGetData import Data_Getter_Pred
from PredictFromModel import prediction
from wsgiref import simple_server
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

@app.route('/',methods=['GET'])
@cross_origin()
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    if request.form is not None:
        age=float(request.form['age'])
        education = float(request.form['education'])
        capital_gain = int(request.form['capital_gain'])
        capital_loss = int(request.form['capital_loss'])
        hours_per_week = int(request.form['hours_per_week'])
        if(request.form['workclass']=="Federal_gov"):
            workclass_Federal_gov=1
            workclass_Local_gov=0
            workclass_Never_worked=0
            workclass_Private=0
            workclass_Self_emp_inc=0
            workclass_Self_emp_not_inc=0
            workclass_State_gov=0
            workclass_Without_pay=0
        else:
            if (request.form['workclass'] == "Local_gov"):
                workclass_Federal_gov = 0
                workclass_Local_gov = 1
                workclass_Never_worked = 0
                workclass_Private = 0
                workclass_Self_emp_inc = 0
                workclass_Self_emp_not_inc = 0
                workclass_State_gov = 0
                workclass_Without_pay = 0
            else:
                if (request.form['workclass'] == "Never_worked"):
                    workclass_Federal_gov = 0
                    workclass_Local_gov = 0
                    workclass_Never_worked = 1
                    workclass_Private = 0
                    workclass_Self_emp_inc = 0
                    workclass_Self_emp_not_inc = 0
                    workclass_State_gov = 0
                    workclass_Without_pay = 0
                else:
                    if (request.form['workclass'] == "Private"):
                        workclass_Federal_gov = 0
                        workclass_Local_gov = 0
                        workclass_Never_worked = 0
                        workclass_Private = 1
                        workclass_Self_emp_inc = 0
                        workclass_Self_emp_not_inc = 0
                        workclass_State_gov = 0
                        workclass_Without_pay = 0
                    else:
                        if (request.form['workclass'] == "Self_emp_inc"):
                            workclass_Federal_gov = 0
                            workclass_Local_gov = 0
                            workclass_Never_worked = 0
                            workclass_Private = 0
                            workclass_Self_emp_inc = 1
                            workclass_Self_emp_not_inc = 0
                            workclass_State_gov = 0
                            workclass_Without_pay = 0
                        else:
                            if (request.form['workclass'] == "Self_emp_not_inc"):
                                workclass_Federal_gov = 0
                                workclass_Local_gov = 0
                                workclass_Never_worked = 0
                                workclass_Private = 0
                                workclass_Self_emp_inc = 0
                                workclass_Self_emp_not_inc = 1
                                workclass_State_gov = 0
                                workclass_Without_pay = 0
                            else:
                                if (request.form['workclass'] == "State_gov"):
                                    workclass_Federal_gov = 0
                                    workclass_Local_gov = 0
                                    workclass_Never_worked = 0
                                    workclass_Private = 0
                                    workclass_Self_emp_inc = 0
                                    workclass_Self_emp_not_inc = 0
                                    workclass_State_gov = 1
                                    workclass_Without_pay = 0
                                else:
                                    workclass_Federal_gov = 0
                                    workclass_Local_gov = 0
                                    workclass_Never_worked = 0
                                    workclass_Private = 0
                                    workclass_Self_emp_inc = 0
                                    workclass_Self_emp_not_inc = 0
                                    workclass_State_gov = 0
                                    workclass_Without_pay = 1

        if(request.form['marital_status']=="Married_AF_spouse"):
            marital_status_Married_AF_spouse=1
            marital_status_Married_civ_spouse=0
            marital_status_Married_spouse_absent=0
            marital_status_Never_married=0
            marital_status_Separated=0
            marital_status_Widowed=0
        else:
            if (request.form['marital_status'] == "Married_civ_spouse"):
                marital_status_Married_AF_spouse = 0
                marital_status_Married_civ_spouse = 1
                marital_status_Married_spouse_absent = 0
                marital_status_Never_married = 0
                marital_status_Separated = 0
                marital_status_Widowed = 0
            else:
                if (request.form['marital_status'] == "Married_spouse_absent"):
                    marital_status_Married_AF_spouse = 0
                    marital_status_Married_civ_spouse = 0
                    marital_status_Married_spouse_absent = 1
                    marital_status_Never_married = 0
                    marital_status_Separated = 0
                    marital_status_Widowed = 0
                else:
                    if (request.form['marital_status'] == "Never_married"):
                        marital_status_Married_AF_spouse = 0
                        marital_status_Married_civ_spouse = 0
                        marital_status_Married_spouse_absent = 0
                        marital_status_Never_married = 1
                        marital_status_Separated = 0
                        marital_status_Widowed = 0
                    else:
                        if (request.form['marital_status'] == "Separated"):
                            marital_status_Married_AF_spouse = 0
                            marital_status_Married_civ_spouse = 0
                            marital_status_Married_spouse_absent = 0
                            marital_status_Never_married = 0
                            marital_status_Separated = 1
                            marital_status_Widowed = 0
                        else:
                            marital_status_Married_AF_spouse = 0
                            marital_status_Married_civ_spouse = 0
                            marital_status_Married_spouse_absent = 0
                            marital_status_Never_married = 0
                            marital_status_Separated = 0
                            marital_status_Widowed = 1
        if(request.form['occupation'] =="Adm_clerical"):
            occupation_Adm_clerical = 1
            occupation_Armed_Forces = 0
            occupation_Craft_repair = 0
            occupation_Exec_managerial = 0
            occupation_Farming_fishing = 0
            occupation_Handlers_cleaners = 0
            occupation_Machine_op_inspct = 0
            occupation_Other_service = 0
            occupation_Priv_house_serv = 0
            occupation_Prof_specialty = 0
            occupation_Protective_serv = 0
            occupation_Sales = 0
            occupation_Tech_support = 0
            occupation_Transport_moving = 0
        else:
            if (request.form['occupation'] == "Armed_Forces"):
                occupation_Adm_clerical = 0
                occupation_Armed_Forces = 1
                occupation_Craft_repair = 0
                occupation_Exec_managerial = 0
                occupation_Farming_fishing = 0
                occupation_Handlers_cleaners = 0
                occupation_Machine_op_inspct = 0
                occupation_Other_service = 0
                occupation_Priv_house_serv = 0
                occupation_Prof_specialty = 0
                occupation_Protective_serv = 0
                occupation_Sales = 0
                occupation_Tech_support = 0
                occupation_Transport_moving = 0
            else:
                if (request.form['occupation'] == "Craft_repair"):
                    occupation_Adm_clerical = 0
                    occupation_Armed_Forces = 0
                    occupation_Craft_repair = 1
                    occupation_Exec_managerial = 0
                    occupation_Farming_fishing = 0
                    occupation_Handlers_cleaners = 0
                    occupation_Machine_op_inspct = 0
                    occupation_Other_service = 0
                    occupation_Priv_house_serv = 0
                    occupation_Prof_specialty = 0
                    occupation_Protective_serv = 0
                    occupation_Sales = 0
                    occupation_Tech_support = 0
                    occupation_Transport_moving = 0
                else:
                    if (request.form['occupation'] == "Exec_managerial"):
                        occupation_Adm_clerical = 0
                        occupation_Armed_Forces = 0
                        occupation_Craft_repair = 0
                        occupation_Exec_managerial = 1
                        occupation_Farming_fishing = 0
                        occupation_Handlers_cleaners = 0
                        occupation_Machine_op_inspct = 0
                        occupation_Other_service = 0
                        occupation_Priv_house_serv = 0
                        occupation_Prof_specialty = 0
                        occupation_Protective_serv = 0
                        occupation_Sales = 0
                        occupation_Tech_support = 0
                        occupation_Transport_moving = 0
                    else:
                        if (request.form['occupation'] == "Farming_fishing"):
                            occupation_Adm_clerical = 0
                            occupation_Armed_Forces = 0
                            occupation_Craft_repair = 0
                            occupation_Exec_managerial = 0
                            occupation_Farming_fishing = 1
                            occupation_Handlers_cleaners = 0
                            occupation_Machine_op_inspct = 0
                            occupation_Other_service = 0
                            occupation_Priv_house_serv = 0
                            occupation_Prof_specialty = 0
                            occupation_Protective_serv = 0
                            occupation_Sales = 0
                            occupation_Tech_support = 0
                            occupation_Transport_moving = 0
                        else:
                            if (request.form['occupation'] == "Handlers_cleaners"):
                                occupation_Adm_clerical = 0
                                occupation_Armed_Forces = 0
                                occupation_Craft_repair = 0
                                occupation_Exec_managerial = 0
                                occupation_Farming_fishing = 0
                                occupation_Handlers_cleaners = 1
                                occupation_Machine_op_inspct = 0
                                occupation_Other_service = 0
                                occupation_Priv_house_serv = 0
                                occupation_Prof_specialty = 0
                                occupation_Protective_serv = 0
                                occupation_Sales = 0
                                occupation_Tech_support = 0
                                occupation_Transport_moving = 0
                            else:
                                if (request.form['occupation'] == "Machine_op_inspct"):
                                    occupation_Adm_clerical = 0
                                    occupation_Armed_Forces = 0
                                    occupation_Craft_repair = 0
                                    occupation_Exec_managerial = 0
                                    occupation_Farming_fishing = 0
                                    occupation_Handlers_cleaners = 0
                                    occupation_Machine_op_inspct = 1
                                    occupation_Other_service = 0
                                    occupation_Priv_house_serv = 0
                                    occupation_Prof_specialty = 0
                                    occupation_Protective_serv = 0
                                    occupation_Sales = 0
                                    occupation_Tech_support = 0
                                    occupation_Transport_moving = 0
                                else:
                                    if (request.form['occupation'] == "Other_service"):
                                        occupation_Adm_clerical = 0
                                        occupation_Armed_Forces = 0
                                        occupation_Craft_repair = 0
                                        occupation_Exec_managerial = 0
                                        occupation_Farming_fishing = 0
                                        occupation_Handlers_cleaners = 0
                                        occupation_Machine_op_inspct = 0
                                        occupation_Other_service = 1
                                        occupation_Priv_house_serv = 0
                                        occupation_Prof_specialty = 0
                                        occupation_Protective_serv = 0
                                        occupation_Sales = 0
                                        occupation_Tech_support = 0
                                        occupation_Transport_moving = 0
                                    else:
                                        if (request.form['occupation'] == "Priv_house_serv"):
                                            occupation_Adm_clerical = 0
                                            occupation_Armed_Forces = 0
                                            occupation_Craft_repair = 0
                                            occupation_Exec_managerial = 0
                                            occupation_Farming_fishing = 0
                                            occupation_Handlers_cleaners = 0
                                            occupation_Machine_op_inspct = 0
                                            occupation_Other_service = 0
                                            occupation_Priv_house_serv = 1
                                            occupation_Prof_specialty = 0
                                            occupation_Protective_serv = 0
                                            occupation_Sales = 0
                                            occupation_Tech_support = 0
                                            occupation_Transport_moving = 0
                                        else:
                                            if (request.form['occupation'] == "Prof_specialty"):
                                                occupation_Adm_clerical = 0
                                                occupation_Armed_Forces = 0
                                                occupation_Craft_repair = 0
                                                occupation_Exec_managerial = 0
                                                occupation_Farming_fishing = 0
                                                occupation_Handlers_cleaners = 0
                                                occupation_Machine_op_inspct = 0
                                                occupation_Other_service = 0
                                                occupation_Priv_house_serv = 0
                                                occupation_Prof_specialty = 1
                                                occupation_Protective_serv = 0
                                                occupation_Sales = 0
                                                occupation_Tech_support = 0
                                                occupation_Transport_moving = 0
                                            else:
                                                if (request.form['occupation'] == "Protective_serv"):
                                                    occupation_Adm_clerical = 0
                                                    occupation_Armed_Forces = 0
                                                    occupation_Craft_repair = 0
                                                    occupation_Exec_managerial = 0
                                                    occupation_Farming_fishing = 0
                                                    occupation_Handlers_cleaners = 0
                                                    occupation_Machine_op_inspct = 0
                                                    occupation_Other_service = 0
                                                    occupation_Priv_house_serv = 0
                                                    occupation_Prof_specialty = 0
                                                    occupation_Protective_serv = 1
                                                    occupation_Sales = 0
                                                    occupation_Tech_support = 0
                                                    occupation_Transport_moving = 0
                                                else:
                                                    if (request.form['occupation'] == "Sales"):
                                                        occupation_Adm_clerical = 0
                                                        occupation_Armed_Forces = 0
                                                        occupation_Craft_repair = 0
                                                        occupation_Exec_managerial = 0
                                                        occupation_Farming_fishing = 0
                                                        occupation_Handlers_cleaners = 0
                                                        occupation_Machine_op_inspct = 0
                                                        occupation_Other_service = 0
                                                        occupation_Priv_house_serv = 0
                                                        occupation_Prof_specialty = 0
                                                        occupation_Protective_serv = 0
                                                        occupation_Sales = 1
                                                        occupation_Tech_support = 0
                                                        occupation_Transport_moving = 0
                                                    else:
                                                        if (request.form['occupation'] == "Tech_support"):
                                                            occupation_Adm_clerical = 0
                                                            occupation_Armed_Forces = 0
                                                            occupation_Craft_repair = 0
                                                            occupation_Exec_managerial = 0
                                                            occupation_Farming_fishing = 0
                                                            occupation_Handlers_cleaners = 0
                                                            occupation_Machine_op_inspct = 0
                                                            occupation_Other_service = 0
                                                            occupation_Priv_house_serv = 0
                                                            occupation_Prof_specialty = 0
                                                            occupation_Protective_serv = 0
                                                            occupation_Sales = 0
                                                            occupation_Tech_support = 1
                                                            occupation_Transport_moving = 0
                                                        else:
                                                            occupation_Adm_clerical = 0
                                                            occupation_Armed_Forces = 0
                                                            occupation_Craft_repair = 0
                                                            occupation_Exec_managerial = 0
                                                            occupation_Farming_fishing = 0
                                                            occupation_Handlers_cleaners = 0
                                                            occupation_Machine_op_inspct = 0
                                                            occupation_Other_service = 0
                                                            occupation_Priv_house_serv = 0
                                                            occupation_Prof_specialty = 0
                                                            occupation_Protective_serv = 0
                                                            occupation_Sales = 0
                                                            occupation_Tech_support = 0
                                                            occupation_Transport_moving = 1
        if(request.form['relationship']=="Not_in_family"):
            relationship_Not_in_family=1
            relationship_Other_relative=0
            relationship_Own_child=0
            relationship_Unmarried=0
            relationship_Wife=0
        else:
            if (request.form['relationship'] == "Other_relative"):
                relationship_Not_in_family = 0
                relationship_Other_relative = 1
                relationship_Own_child = 0
                relationship_Unmarried = 0
                relationship_Wife = 0
            else:
                if (request.form['relationship'] == "Own_child"):
                    relationship_Not_in_family = 0
                    relationship_Other_relative = 0
                    relationship_Own_child = 1
                    relationship_Unmarried = 0
                    relationship_Wife = 0
                else:
                    if (request.form['relationship'] == "Unmarried"):
                        relationship_Not_in_family = 0
                        relationship_Other_relative = 0
                        relationship_Own_child = 0
                        relationship_Unmarried = 1
                        relationship_Wife = 0
                    else:
                        relationship_Not_in_family = 0
                        relationship_Other_relative = 0
                        relationship_Own_child = 0
                        relationship_Unmarried = 0
                        relationship_Wife = 1
        if(request.form['race']=="Asian_Pac_Islander"):
            race_Asian_Pac_Islander=1
            race_Black=0
            race_Other=0
            race_White=0
        else:
            if (request.form['race'] == "Black"):
                race_Asian_Pac_Islander = 0
                race_Black = 1
                race_Other = 0
                race_White = 0
            else:
                if (request.form['race'] == "Other"):
                    race_Asian_Pac_Islander = 0
                    race_Black = 0
                    race_Other = 1
                    race_White = 0
                else:
                        race_Asian_Pac_Islander = 0
                        race_Black = 0
                        race_Other = 0
                        race_White = 1
        if(request.form['gender']=="female"):
            sex_Male=0
        else:
            sex_Male=1
        if(request.form['country']=="United States"):
            country_UnitedStates=1
        else:
            country_UnitedStates = 0


        model_load = finder.ModelFinder()
        model = model_load.load_model()
        var = model.get_booster().feature_names
        to_predict_list=[age,capital_gain,capital_loss,hours_per_week,workclass_Federal_gov,workclass_Local_gov,workclass_Never_worked,workclass_Private,workclass_Self_emp_inc,workclass_Self_emp_not_inc,workclass_State_gov,workclass_Without_pay,marital_status_Married_AF_spouse,marital_status_Married_civ_spouse,marital_status_Married_spouse_absent,marital_status_Never_married,marital_status_Separated,marital_status_Widowed,occupation_Adm_clerical,occupation_Armed_Forces,occupation_Craft_repair,occupation_Exec_managerial,occupation_Farming_fishing,occupation_Handlers_cleaners,occupation_Machine_op_inspct,occupation_Other_service,occupation_Priv_house_serv,occupation_Prof_specialty,occupation_Protective_serv,occupation_Sales,occupation_Tech_support,occupation_Transport_moving,relationship_Not_in_family,relationship_Other_relative,relationship_Own_child,relationship_Unmarried,relationship_Wife,race_Asian_Pac_Islander,race_Black,race_Other,race_White,sex_Male,country_UnitedStates,education]
        to_predict = np.array(to_predict_list).reshape(1,44)
        to=pd.DataFrame(to_predict,columns=list(var))
        result=model.predict(to)
        if result==0:
            result="<=50K"
        else:
            result=">50K"

        return render_template('home.html', prediction_text="Income is {}".format(result))


    '''except ValueError:
        return render_template('index.html', prediction_text="Error Occurred! %s" %ValueError)
        #print("Error Occurred! %s" %ValueError)
    except KeyError:
        return render_template('index.html', prediction_text="Error Occurred! %s" % KeyError)
        #print(("Error Occurred! %s" %KeyError))
    except Exception as e:
        return render_template('index.html', prediction_text="Error Occurred! %s" % e)
        #print(("Error Occurred! %s" %e))'''




#@app.route('/train',methods=['POST'])
#@cross_origin()
def train_predict():
    try:
        readdataobj = ReadData.ReadData()
        data=readdataobj.readData()
        #datasplitobj=DataSplitting(data)
        #X_train, X_test, y_train, y_test = datasplitobj.split_data()

        preprocess=Preprocessing.Preprocessor()
        data=preprocess.removeExtraSpace(data)
        X, y = preprocess.seperateDependentIndependentColumns(data)
        y=y.map({'<=50K':0,'>50K':1})
        columns_with_null_values , is_null_present=preprocess.columnsWithMissingVlaue(X)
        if(is_null_present):
            X=preprocess.imputeMissingValue(columns_with_null_values,X)
        preprocess.removeUnwantedFeatures(X , ['education-num','fnlwgt'])
        country=np.where(X['country']=="United-States", "UnitedStates", "Non UnitedStates")
        X = preprocess.outliarsCompute(X, 'hours-per-week')
        X['country']=country
        df_numeric=preprocess.scaleDownNumericFeatures(X)
        print("Countries",X['country'].unique())
        df_category=preprocess.encodeCategoryFeatures(X)
        df=pd.concat([df_numeric,df_category], axis=1)
        print(df_category)
        df.to_csv('test.csv')
        X,y = preprocess.handleImbalancedDataSet(df,y)
        split_data = DataSplitting()
        X_train, X_test, y_train, y_test = split_data.split_data(X,y)
        modelfinder=ModelFinder()
        modelfinder.get_best_model(X_train, X_test, y_train, y_test)
    except ValueError:
        print("Value error occurred %s" %ValueError)

    except KeyError:
        print("Key error occurred %s" %KeyError)

    except Exception as e:
        print("Error Occurred while training %s" %e)

    print("Training Successfull !!")

#port = int(os.getenv("PORT",5000))
if __name__=='__main__':
    app.run(debug=True,threaded=True)
    #train_predict()

