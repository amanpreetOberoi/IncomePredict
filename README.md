
# Adult Census Income Prediction

This project aims to show the usage of machine learning and data science techniques in providing a solution to the income equality problem

## Appendix
- [Acknowledgements](Acknowledgements)
- [About_Data](About Data)
- [Motivation](Motivation)
- [Workflow](Workflow)
- [Technologies_Used ](Technologies Used )
- [Demo](Demo)
- [Environment Variables](Environment Variables)
- [Deployment on Heroku](Deployment on Heroku)
- [Documentation](Documentation)
- [Authors](Authors)
- [FAQ](FAQ)


## Acknowledgements

 - The prominent inequality of wealth and income is a huge concern especially in the World. The likelihood of diminishing poverty is one valid reason to reduce the world's surging level of economic inequality. The principle of universal moral equality ensures sustainable development and improve the economic stability of a nation. Governments in different countries have been trying their best to address this problem and provide an optimal solution. This study aims to show the usage of machine learning and data mining techniques in providing a solution to the income equality problem. The UCI Adult Dataset has been used for the purpose. Classification has been done to predict whether a person's yearly income in US falls in the income category of either greater than 50K Dollars or less equal to 50K Dollars category based on a certain set of attributes. 
  
## About Data
#### Attribute Information:

Listing of attributes:

>50K, <=50K.

- age: continuous.
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- fnlwgt: continuous.
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- education-num: continuous.
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Female, Male.
- capital-gain: continuous.
- capital-loss: continuous.
- hours-per-week: continuous.
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

## Motivation

#### Product Perspective
The Adult Census Income Prediction, classified  the income category of either greater than 50K Dollars or less equal to 50K Dollars category of the person by using classification based Supervised Machine Learning algorithms.
#### Problem statement
The Goal is to predict whether a person has an income of more than 50K a year or not. This is basically a binary classification problem where a person is classified into the >50K group or <=50K group.
#### Proposed Solution
The solution here is a classification based Supervised Machine Learning model. It can be implemented by different classification algorithms (like Logistic Regression, Random Forest Classification, Decision Tree Classification, SVC, Xg-boost Classifier, Gausian NB and so on.).

Here first, we are performing Data preprocessing step, in which feature engineering, feature selection, feature scaling steps are performed and then we are going to build model.


## Workflow

#### Performance
Solution of Adult Census Income Prediction is used to classified into the >50K group or <=50K group in advance, so it   should be as accurate as possible so that it should give as much as possible accurate Income prediction.

That’s why before building this model we followed complete process of Machine Learning. Here are summary of complete process:

1.	First we cleaned our dataset properly by removing all extra space present and     duplicate value present in dataset.
2.	Separate the dependent and independent variables.
3.	according to first null values analysis there is no any missing value present in dataset, but take insight look in data  we can see that a symbol '?' present in dataset which means that this is present at the place of  missing value, so we need to replace all these '?' with 'null' values
4.	there is missing values are present in columns 'work-class", "occupation" & "country", all these columns are categorical columns so we  impute the missing values with Mode.

5.	Then we compute all the outliers present in dataset and handle them.

6.	Then we handled categorical variable by performing One-Hot encoding rather than country columns, in country 98% of data refers to United States, so we differentiate data into two entities USA and Non-USA.   .

7.	The data is highly imbalanced then we balanced the dataset by using standard scaler.

8.	Then we split the whole data set train-test split. And  split into  X_train, X_test, y-train and y_test.  
9.	After performing above step I was ready for model training. In this step, I trained my dataset on different classification based supervised Machine Learning Algorithm (Logistic Regressions, Random-Forest Classification, XGBoost Classifier and Gaussian NB). After training the dataset on different algorithms I got highest   accuracy of 86% on XGBoost Classifier

10.	After that I applied hyper-parameter tuning on all model which I have described above. Here also I got highest accuracy of 90% on test dataset by    same XGBoost Classifier.
    ```bash
       		    precision    recall  f1-score   support

               0       0.91      0.88      0.89      8135
               1       0.88      0.91      0.90      8181

        accuracy                           0.90     16316
        macro avg      0.90      0.90      0.90     16316
        weighted avg   0.90      0.90      0.90     16316


    ```
11.	After that I saved my model in pickle file format for model deployment.

12.	After that my model was ready to deploy. I deployed this model on Heroku which is used as a platform as a service (PaaS) to build, run, and operate applications entirely in the cloud.


#### Re-usability
We have done programming of this project in Modular Fashion in which various classes are made so that it should be reusable. So that anyone can add and contribute without facing any problems.

## Technologies Used 

##### Python	
high-level computer programming language used to develop the project 
##### Py-Charm	
an integrated development IDE used in computer programming, for the Python language
##### Pandas	
Open source data analysis and manipulation tool, for the Python programming language.
##### Numpy	
Python library used for working with arrays
##### Matplotlib/Seaborn	
For data visualization and graphical plotting library for Python
##### Scikit-Learn	
Machine learning library used for the Python programming language. It features various classification, regression and clustering algorithms
##### Flask	
A web applications  framework, it's a Python module that lets you develop web applications easily
##### HTML/CSS	
Are two of the core technologies used for building Web pages. HTML provides the structure of the page, CSS the layout for a variety of devices.
##### Heroku	
Is used as a platform as a service (PaaS) that build, run, and operate applications entirely in the cloud 
##### GitHub	
Web-based interface that used Git for the open source version control system 








## Demo
To run application on personal devices :

Application link
[http://incomeprediction190.herokuapp.com](http://incomeprediction190.herokuapp.com)


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file :

The Code is written in Python 3.7. If you don't have Python installed you can find it here. If you are using a lower version of Python you can upgrade using the pip package, 
ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after cloning the repository:
 [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:

For pip installation:

```bash
pip install -r requirements.txt
```
For anaconda installation

```bash
conda env create -f environment.yml
```
For more refrence regarding Managing Anconda environment [click here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)


  
## Deployement on Heroku
Login or signup in order to create virtual app. You can either connect your github profile or download ctl to manually deploy this project.
[![](https://i.imgur.com/dKmlpqX.png)](https://heroku.com)
Our next step would be to follow the instruction given on [Heroku Documentation](https://devcenter.heroku.com/articles/getting-started-with-python) to deploy a web app.

Application Link
[http://incomeprediction190.herokuapp.com](http://incomeprediction190.herokuapp.com/)

  
## Documentation

[Low Level Desgine Document](https://drive.google.com/file/d/1LKudKNWMj-kdNk0MwOPwwQZgn7pBOjmu/view?usp=sharing)

Low-level design (LLD) is a component-level design process that follows a step-by-step refinement process. This process can be used for designing data structures, required application architecture, source code and ultimately, performance algorithms.

[High  Level Desgine Document](https://drive.google.com/file/d/1YjwqbCA1YUiulrj3auOqiV7gSi20u5ji/view?usp=sharing)

The purpose of this High-Level Design (HLD) Document is to add the important         details about this project. Through this HLD Document, I’m going to describe every small and big things about this project.

[Detailed Project Report](https://drive.google.com/file/d/1hKwawWykPr7H-AoxAM3Fb-lp8bvSB9rv/view?usp=sharing) 

A DPR is a final, detailed appraisal report on the project and a blue print for its execution and eventual operation. It provide details of the basic programme the roles and responsibilities.

[Wireframe Document](https://drive.google.com/file/d/1bjzlT1fF4XzGNDT80tsraqHFcTIiEmdS/view?usp=sharing)

 wireframe is a blueprint of a page. It lays an outline of the page structure, information hierarchy, and user flow through the application. 
## Authors
#### Lovepreet Singh
- jsjosan3@gmail.com
- [GitHub](https://www.https://github.com/jsjosan3)
- [Linkedin](https://www.linkedin.com/in/lovepreet-singh-189593154/)
#### Amanpreet Oberoi
- apo28june@gmail.com 
- [GitHib](https://github.com/amanpreetOberoi)
- [Linkedin](https://www.linkedin.com/in/amanpreet-oberoi-031858160/)

## FAQ

#### Q1) What’s the source of data?
 The data for training is taken from the internship portal of ineuron.ai., but along with that data is also avialable on Kaggle.com

####  Q 2) What was the type of data?
 The data was the combination of numerical and Categorical values.

#### Q 3) What’s the complete flow you followed in this Project?
 Refer slide 4th for better Understanding

#### Q 4) How logs are managed?
 We are using different logs as per the steps that we follow invalidation and  modeling like File validation log , Data 	Insertion Model Training log , prediction log  etc.

#### Q 6) What techniques were you using for data pre-processing?
- Removing unwanted attributes
- Visualizing relation of independent variables with each other and output variables
- Checking and changing Distribution of continuous values
- Removing outliers
- Cleaning data and imputing if null values are present.
- Converting categorical data into numeric values.
- Scaling the data
#### Q 7) what kind of coding statderds are used?
coding is done in modulara fasion as per python [Python Developer's Guide](https://www.python.org/dev/)



  
