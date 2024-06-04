# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:01:00 2024

@author: Samane
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
pio.templates.default = "plotly_white"
pio.renderers.default='browser'
data=pd.read_csv("Credit Score Data/train.csv")
#print(data.head())
#print(data.info())

## check the dataset if has any null values##
#print(data.isnull().sum())

##show the number of credit_score values##
#print(data["Credit_Score"].value_counts())

##Credit Scores Based on Occupation##

#fig=px.box(data, x="Occupation", color="Credit_Score",title="Credit Scores Based on Occupation",
#           color_discrete_map={'Good':'green','Standard':'yellow','Poor':'red'})

#fig.show()

## Credit_Mix column transform into numerical feature ##
data['Credit_Mix']=data['Credit_Mix'].map({'Standard':1, 'Good':2,'Bad':0})

## split the data into features and labels which are important ##
x=np.array(data[["Annual_Income", "Monthly_Inhand_Salary", 
                    "Num_Bank_Accounts", "Num_Credit_Card", 
                    "Interest_Rate", "Num_of_Loan", 
                    "Delay_from_due_date", "Num_of_Delayed_Payment", 
                    "Credit_Mix", "Outstanding_Debt", 
                    "Credit_History_Age", "Monthly_Balance"]])
y=np.array(data[['Credit_Score']])

## train ##
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.33,random_state=42)
model= RandomForestClassifier()
model.fit(xtrain, ytrain)

## make prediction ##
print("Credit Score Prediction : ")
a = float(input("Annual Income:"))
b = float(input("Monthly_Inhand_Salary:"))
c = float(input("Number of Bank Accounts:"))
d = float(input("Number of Credit cards:"))
e = float(input("Interest rate: "))
f = float(input("Number of Loans:"))
g = float(input("Average number of days delayed by the person:"))
h = float(input("Number of delayed payments: "))
i = input("Credit Mix (Bad: 0, Standard: 1, Good: 3) :  ")
j = float(input("Outstanding Debt:"))
k = float(input("Credit History Age:"))
l = float(input("Monthly Balance:"))

features = np.array([[a,b, c, d, e, f, g, h, i, j, k, l]])
print("Predicted Credit Score = ", model.predict(features))
