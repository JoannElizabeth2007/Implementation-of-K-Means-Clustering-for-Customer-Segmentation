# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.import pandas module and import the required data seT.
2.Find the null values and count them.
3.Count number of left values.
4.From sklearn import LabelEncoder to convert string values to numerical values.
5.From sklearn.model_selection import train_test_split.
6.Assign the train dataset and test dataset.
7.From sklearn.tree import DecisionTreeClassifier.
8.Use criteria as entropy.
9.From sklearn import metrics.
10.Find the accuracy of our model and predict the require values.
```
## Program:

Program to implement the K Means Clustering for Customer Segmentation.
Developed by: JOANN ELIZABETH SAMUEL 
RegisterNumber: 212224040139
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(data.iloc[:, 3:5])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

km = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = km.fit_predict(data.iloc[:, 3:5])
data["Cluster"] = y_pred

plt.figure(figsize=(8, 6))
colors = ['red', 'black', 'blue', 'green', 'magenta']
for i in range(5):
    cluster = data[data["Cluster"] == i]
    plt.scatter(cluster["Annual Income (k$)"], cluster["Spending Score (1-100)"], 
                c=colors[i], label=f"Cluster {i}")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.legend()
plt.grid(True)
plt.show()

```

## Output:

## DATA.HEAD()
![image](https://github.com/user-attachments/assets/cecbb6b7-0d53-409e-bae4-b93304865bb8)

## DATA.INF0()
![image](https://github.com/user-attachments/assets/3a258fd1-e9c2-4ef2-927e-5cf9cda73b93)

## DATA.ISNULL().SUM()
![image](https://github.com/user-attachments/assets/04c5f927-0271-490b-a282-fff74859d28c)

## PLOT USING ELBOW METHOD 
![image](https://github.com/user-attachments/assets/826e205f-a3b1-499c-b922-dabac7b91e7b)

## CUSTOMER SEGMENT
![image](https://github.com/user-attachments/assets/3d09e5bb-f87a-4d06-9388-33e77a218304)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
