# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results
   

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: MAGESH BOOPATHI.M
RegisterNumber: 24900855 

```
```python
 import pandas as pd
 data=pd.read_csv("Placement_Data (1).csv")
 print(data.head())
 data1=data.copy()
 data1=data1.drop(["sl_no","salary"],axis=1)
 print(data1.head())
 data1.isnull().sum()
 data1.duplicated().sum()
 from sklearn.preprocessing import LabelEncoder
 le=LabelEncoder()
 data1["gender"]=le.fit_transform(data1["gender"])
 data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
 data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
 data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
 data1["degree_t"]=le.fit_transform(data1["degree_t"])
 data1["workex"]=le.fit_transform(data1["workex"])
 data1["specialisation"]=le.fit_transform(data1["specialisation"])
 data1["status"]=le.fit_transform(data1["status"])
 print(data1)
 x=data1.iloc[:,:-1]
 x
 y=data1["status"]
 y
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
 from sklearn.linear_model import LogisticRegression
 lr=LogisticRegression(solver="liblinear")
 lr.fit(x_train,y_train)
 y_pred=lr.predict(x_test)
 print(y_pred)
 from sklearn.metrics import accuracy_score
 accuracy=accuracy_score(y_test,y_pred)
 print(accuracy)
 from sklearn.metrics import confusion_matrix
 confusion=confusion_matrix(y_test,y_pred)
 print(confusion)
 from sklearn.metrics import classification_report
 classification_report1=classification_report(y_test,y_pred)
 print(classification_report1)
 lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
HEAD:

![Screenshot 2024-11-15 212448](https://github.com/user-attachments/assets/c68ebbf5-bb84-4720-bc2f-c9187f040a2f)

COPY:

![Screenshot 2024-11-15 212502](https://github.com/user-attachments/assets/210768b6-514b-4866-9dc5-c5bd15fe1456)

FIT TRANSFORM:

![Screenshot 2024-11-15 212558](https://github.com/user-attachments/assets/188522d0-8e5e-4673-837c-d34879207dcc)

LOGISTIC  REGRESSION:

![Screenshot 2024-11-15 212642](https://github.com/user-attachments/assets/860f67fe-0c15-4020-8dd8-d16aa24729ce)

ACCURACY SCORE:

![Screenshot 2024-11-15 212659](https://github.com/user-attachments/assets/760fea23-1f13-4b4d-868d-7ae31e5c9508)

CONFUSION MATRIX:

![Screenshot 2024-11-15 213236](https://github.com/user-attachments/assets/54358bb4-c98e-4250-be70-2a25a53f7766)

CLASSIFICATION REPORT:

![Screenshot 2024-11-15 212758](https://github.com/user-attachments/assets/fe458943-1364-40d2-9e3f-91808beb53f8)

PREDICTION:

![Screenshot 2024-11-15 212826](https://github.com/user-attachments/assets/477164f2-c4fe-449f-a643-9dd2e4d71c29)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
