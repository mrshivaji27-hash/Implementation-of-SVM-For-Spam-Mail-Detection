# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Detect File Encoding: Use chardet to determine the dataset's encoding.
Load Data: Read the dataset with pandas.read_csv using the detected encoding.
Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
Train SVM Model: Fit an SVC model on the training data.
Predict Labels: Predict test labels using the trained SVM model.
Evaluate Model: Calculate and display accuracy with metrics.accuracy_score

## Program:
    import chardet
    file='spam.csv'
    with open(file, 'rb') as rawdata:
        result = chardet.detect (rawdata.read(100000))
    result
<br>

    import pandas as pd
    data=pd.read_csv('spam.csv', encoding='Windows-1252')
    data.info()



    data.isnull().sum()


    
    x=data["v1"].values
    y=data["v2"].values




    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
        from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()





    x_train=cv.fit_transform(x_train)
    x_test=cv.transform(x_test)




    from sklearn.svm import SVC
    svc=SVC()
    svc.fit(x_train, y_train)
    y_pred=svc.predict(x_test)
    y_pred



    from sklearn import metrics
    accuracy=metrics.accuracy_score(y_test,y_pred)
    accuracy

## Output:
<img width="708" height="150" alt="398595784-ba42ca2d-85a5-4795-b40d-63df37236a81" src="https://github.com/user-attachments/assets/ed9e1125-ca00-48c7-a2c7-d5adb56ee129" />



<img width="726" height="237" alt="398595804-22edcf48-dd25-4513-867d-6c7430da985d" src="https://github.com/user-attachments/assets/2d4c2817-59ee-4e5c-afc8-08db308a7ab6" />



<img width="414" height="274" alt="398595822-618324db-35cd-4c45-8469-466a22c5cf86" src="https://github.com/user-attachments/assets/0aad7399-b484-479b-89ab-47c15671f84c" />


<br>

<img width="220" height="169" alt="398595845-81bc5619-eb01-41f1-9218-e95a64875d42" src="https://github.com/user-attachments/assets/9f599779-f0c8-4fab-8a4d-74ae0f1fe80e" />


<img width="653" height="184" alt="398595874-6836f245-d141-4e8b-8bfa-e2d8eac18a76" src="https://github.com/user-attachments/assets/f3efc62d-c5ad-44c7-901f-9802d5feeb47" />



<img width="422" height="131" alt="398595912-3da73419-4d11-45bf-b361-b1fd344e732e" src="https://github.com/user-attachments/assets/5e88ac6b-3c89-4c5a-8cbc-b04b4e91bfb8" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
