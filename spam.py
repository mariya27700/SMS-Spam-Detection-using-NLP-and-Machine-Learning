import numpy as np 
import pandas as pd 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
import matplotlib.pyplot as plt

df=pd.read_csv("spam.csv",encoding="latin-1")
df.isnull().sum()
df.fillna("Random value(not necessary)",inplace=True)
# print(df.head())
# print(df.shape)

#text cleaning

#1.lowercase
df["v1"]=df["v1"].str.lower()
df["v2"]=df["v2"].str.lower()

#2,3,4.No special characters and digits and spaces
df["v2"]=df["v2"].str.replace(r"[^\w\s]"," ",regex=True)
df["v2"]=df["v2"].str.replace(r"[\d+]"," ",regex=True)
df["v2"]=df["v2"].str.replace(r"\s+"," ",regex=True)

#tokenization
df["v2"]=df["v2"].str.split()

#stop word removal
s_w = set(stopwords.words('english'))
custom_sw = {
    "u","ur","r","n","nd","fr","b",
    "txt","msg","mobile","reply",
    "å","ì_"
}

s_w.update(custom_sw)
df["v2"]=df["v2"].apply(
    lambda x:[w for w in x if w not in s_w ]
)
print(df["v2"])
#lemmatization
lemmer=WordNetLemmatizer()
df["v2"]=df["v2"].apply(lambda x:" ".join([lemmer.lemmatize(w) for w in x]))

#tfidf
tf=TfidfVectorizer()
X=tf.fit_transform(df["v2"])
y=df["v1"]

#prediction
def prediction(n):
    print("Prediction: ",n[0])

#accuracy,precision,confusion metrics
def apscore(yp):
    print("Accuracy: ",accuracy_score(y_test,yp))
    print("Precision: ",precision_score(y_test,yp,pos_label="spam"))
    print("Confusion metrics: ",confusion_matrix(y_test,yp))

#prediction value
new_email=["Free entry win cash now"]
new_email_vc=tf.transform(new_email)

#training
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)
print("Logistic Regression")
m1=LogisticRegression()
m1.fit(X_train,y_train)
yp1=m1.predict(X_test)
p1=m1.predict(new_email_vc)
p11=precision_score(y_test,yp1,pos_label="spam")
prediction(p1)
apscore(yp1)

print("Naive Bayes")
m2=MultinomialNB()
m2.fit(X_train,y_train)
yp2=m2.predict(X_test)
p2=m2.predict(new_email_vc)
p21=precision_score(y_test,yp2,pos_label="spam")
prediction(p2)
apscore(yp2)


if p11> p21:
    print("Best model is Logistic Regression Model")
elif p11<p21:
    print("Best model is Naive Bayes")
else:
    print("Both model are good(same precision)")

#kmeans

#elbow method
inertia_value=[]
for i in range(1,11):
    model=KMeans(
        n_clusters=i,
        random_state=42
    )
    model.fit(X)
    inertia_value.append(model.inertia_)
plt.plot(
    range(1,11),
    inertia_value
)
plt.show()

model=KMeans(n_clusters=3,random_state=42)
model.fit(X)
cluster=model.predict(X)
df["cluster"]=cluster
c1=model.predict(new_email_vc)

print("Prediction of Clustering: ",c1[0])







