# Numpy and pandas for mathematical operations
import numpy as np
import pandas as pd

# To read csv dataset files
import csv

# Regular expression, for pattern matching
import re

# The preprocessing module provides functions for data preprocessing tasks such as scaling and handling missing data.
from sklearn import preprocessing

# For Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# train-test split
from sklearn.model_selection import train_test_split

# For building decision tree models, and _tree to access low-level decision of tree structure
from sklearn.tree import DecisionTreeClassifier, _tree

# For evaluating model performance using cross_validation
from sklearn.model_selection import cross_val_score

#Import Support Vector Classification from sklearn library for model deployment
from sklearn.svm import SVC

# Remove unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


#--------------------------------------
#Exploratory Data Analysis
#--------------------------------------


training = pd.read_csv('Data_training.csv')
testing = pd.read_csv('Data_testing.csv')

shape = training.shape
#print("Shape of Training dataset: ", shape)

description = training.describe()
#print(description) this shows each column of the training file and their description (mean, standard deviation, min, 25...)

#info_df = training.info()
#print(info_df) shows information about the dataset

null_values_count = training.isnull().sum()
#print(null_values_count)  shows number of null input in the data table

#Displays all columns
#pd.set_option('display.max_columns', None)

# print(training.head(8))

cols = training.columns
cols = cols[:-1]

#x is features/symptoms
x = training[cols]

#y is target/prognosis
y = training['prognosis']



#Visualizes the distribution of the prognosis target variable to see if there is class imbalance. This distribution can influence model accuracy.
plt.figure(figsize=(10, 20))
sns.countplot(y='prognosis', data=training)
plt.title('Distribution of Target (Prognosis)')
#plt.show()

# Grouping Data by Prognosis and Finding Maximum Values
reduced_data = training.groupby(training['prognosis']).max()
# Display the first five rows of the reduced data
reduced_data.head()


#--------------------------------------
#Data Pre-processing
#--------------------------------------


#converts categorical strings to numerical values
le = preprocessing.LabelEncoder()

#fit the label encoder to the 'target' which is the prognosis
le.fit(y)
y = le.transform(y)

# Splitting the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Features for testing except the last variable
testx    = testing[cols]

# Target variable for Testing
testy    = testing['prognosis']

# Transforming categorical value into numerical labels
testy    = le.transform(testy)


#--------------------------------------
#Model Building and Evaluation
#--------------------------------------

#Implement decision tree model
clf1  = DecisionTreeClassifier()

#Fit the training data
clf = clf1.fit(x_train, y_train)

#Cross validation to assess model performance by evaluating on three different splits
scores = cross_val_score(clf, x_test, y_test, cv=3)
# print("Mean Score: ", scores.mean())

# initializes an SVM model, fits it to the training data, and calculates accuracy on the test set
model = SVC()
model.fit(x_train, y_train)
# print("Accuracy score for svm: ", model.score(x_test, y_test))


# identifies the most significant features using the trained decision tree classifier
# Sort indices in descending order based on feature importance

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

# Initialize dictionaries to store symptom severity, description, and precautions
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()
symptoms_dict = {}

# Populate symptoms dictionary with indices
for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index

# Function to calculate the overall severity of the symptom
def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        print("\nYou should take the consultation from doctor. ")
    else:
        print("\nIt might not be that bad but you should take precautions.")


# Function to read and store symptom descriptions from a CSV file
def getDescription():
    global description_list
    with open('Data_symptom_description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)


# Function to read and store symptom severity information from a CSV file
def getSeverityDict():
    global severityDictionary
    with open('Data_symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


# Function to read and store symptom precaution information from a CSV file
def getprecautionDict():
    global precautionDictionary
    with open('Data_symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)

def getInfo():
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    print("\nYour Name? \t\t\t\t\t\t\t\t\t\t",end="-> ")
    name=input("")
    print("Hello", name)


# Find matching items in a list of diseases
def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')

    #create a regular expression pattern
    patt = f"{inp}"

    #compiled regex object based on this pattern
    regexp = re.compile(patt)

    #Search for Matches in the Disease List
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]

# predict a disease based on symptoms provided by the user using Decision Tree
def sec_predict(symptoms_exp):
    df = pd.read_csv('Data_training.csv')

    #contains all symptoms except for last one
    X = df.iloc[:, :-1]

    # contains all prognosis
    y = df['prognosis']

    # splits the dataset into training (70%) and testing (30%) subsets, helping to evaluate model performance.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

    # initialize and train the decision tree
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    # Map Symptom Names to Indices
    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}

    # creates zeros with the same length of as the number of symptoms
    input_vector = np.zeros(len(symptoms_dict))

    # For each symptom in the user's symptoms, the corresponding position is set to 1 to indicate its process
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    # Uses the trained decision tree to predict the disease based on the user’s symptoms
    return rf_clf.predict([input_vector])


# Interprets and returns the names of diseases based on the encoded numerical values in a node from the trained model
def print_disease(node):
    node = node[0]
    val  = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))



def tree_to_code(tree, feature_names):

    # extracting the Tree Structure
    tree_ = tree.tree_

    # creat list that maps the feature indices to their corresponding names
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    while True:

      # Prompt the user to enter the symptom
        print("\nEnter the symptom you are experiencing  \t\t",end="-> ")
        disease_input = input("")

        # checks validity of user input
        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        if conf==1:
            print("searches related to input: ")
            for num,it in enumerate(cnf_dis):
                print(num, ")",it)
            if num!=0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp=0

            disease_input=cnf_dis[conf_inp]
            break
        else:
            print("Enter valid symptom.")

    # asks how long the user has been experiencing symptoms
    while True:
        try:
            num_days=int(input("\nFor how many days ? : "))
            break
        except:
            print("Enter valid input.")

    # navigate the decision tree based on the user’s input
    # checks the features at the current node against the input, determining whether to follow the left or right child nodes
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])

            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

            print("\nAre you experiencing any ")
            symptoms_exp=[]
            for syms in list(symptoms_given):
                inp=""
                print(syms+"? [yes/no]: ",end='')
                while True:
                    inp=input("")
                    if(inp=="yes" or inp=="no"):
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ",end="")
                if(inp=="yes"):
                    symptoms_exp.append(syms)

            # Get a second prediction based on the symptoms
            second_prediction=sec_predict(symptoms_exp)
            # print(second_prediction)

            # calculates severity
            calc_condition(symptoms_exp,num_days)

            # if both predictions are the same, then print one disease
            if(present_disease[0]==second_prediction[0]):
                print("\nYou may have ", present_disease[0])
                print(description_list[present_disease[0]])

            # if predictions are not the same, then print both
            else:
                print("\nYou may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            # print the precaution list based on the disease
            precution_list=precautionDictionary[present_disease[0]]
            print("\nTake following measures : ")
            for  i,j in enumerate(precution_list):
                print(i+1,")",j)


    recurse(0, 1)


getSeverityDict()
getDescription()
getprecautionDict()
getInfo()
tree_to_code(clf, cols)
print(
    "----------------------------------------------------------------------------------------------------------------------------------")
