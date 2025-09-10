# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: Tejasree.K

RegisterNumber:  212224240168
*/
```

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_csv("/content/Placement_Data (1).csv")
print("Placement Data:")
print(data)
if 'salary' in data.columns:
    print("\nSalary Data:")
    print(data['salary'])
else:
    print("\n'Salary' column not found in DataFrame")
data1 = data.drop(["salary"], axis=1, errors='ignore')
print("\nMissing Values Check:")
print(data1.isnull().sum())
print("\nDuplicate Rows Check:")
print(data1.duplicated().sum())

print("\nCleaned Data:")
print(data1)
le = LabelEncoder()

categorical_columns = ['workex', 'status', 'hsc_s', 'gender']  
for column in categorical_columns:
    if column in data1.columns:
        data1[column] = le.fit_transform(data1[column])
    else:
        print(f"'{column}' column not found in DataFrame")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
x = pd.get_dummies(x, drop_first=True)  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_report1 = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("Classification Report:\n", classification_report1)
print("\nY Prediction Array:")
print(y_pred)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()

*/
```

## Output:

Placement data:

<img width="1035" height="721" alt="image" src="https://github.com/user-attachments/assets/a1f8922a-e6fc-492a-9c45-ff73c53de81f" />

Salary Data :

<img width="586" height="363" alt="image" src="https://github.com/user-attachments/assets/4dd5f3ff-f928-4014-9f7a-932c70448d72" />

Missing value check:

<img width="412" height="509" alt="image" src="https://github.com/user-attachments/assets/1df6bbdb-a19b-4749-b28b-23c44752ed52" />

Cleaned data:

<img width="1035" height="802" alt="image" src="https://github.com/user-attachments/assets/0b4d138e-43be-4bea-9414-1dac5594f725" />

Y prediction array:

<img width="946" height="136" alt="image" src="https://github.com/user-attachments/assets/3d18ab42-ce63-47ec-839b-a2dc57e0a4e6" />

Accuracy value:

<img width="378" height="37" alt="image" src="https://github.com/user-attachments/assets/fea924b1-bcdd-4bbb-bbeb-398d1359525f" />

Confusion Matix:

<img width="1001" height="799" alt="image" src="https://github.com/user-attachments/assets/b6dc4832-e85a-48f4-859e-29170354b677" />


Classification Report :

<img width="786" height="248" alt="image" src="https://github.com/user-attachments/assets/b7c12332-38e5-41b2-bec4-65c5e7b404bf" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
