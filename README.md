# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Collect a labeled dataset of emails, distinguishing between spam and non-spam.

2.Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.

3.Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF.

4.Split the dataset into a training set and a test set.

5.Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.

6.Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.

7.Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.

8.Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.

9.Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed.

## Program:
```py
Program to implement the SVM For Spam Mail Detection..
Developed by: MIDHUN AZHAHU RAJA P
RegisterNumber: 212222240066


import chardet
file = "spam.csv"
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()

data.info()

data.isnull().sum()

x = data["v1"].values
y = data["v2"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy
```

## Output:

## RESULT OUTPUT:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393818/ec348ff3-70a2-4fea-b4e2-06dd8e7fba17)

### data.head():

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393818/5179fd57-b824-40dc-8d5c-1565dec7a23b)

### data.info:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393818/bebaa446-20f3-418f-9ad0-667bb4c50ee4)

### data.isnull().sum():

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393818/ef8834d9-b420-4179-bb21-d37b5433405f)

### Y_prediction value:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393818/d4071808-9de5-4f14-8121-bbda73d80ca1)

### Accuracy value:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393818/53de0f17-3ce2-4f5c-a232-0f4e4fd93655)








## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
