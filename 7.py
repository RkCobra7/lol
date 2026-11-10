import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score,f1_score

train = pd.read_csv("20news_train.csv")
test = pd.read_csv("20news_test.csv")

x_train_text = train['text'].fillna('')
y_train = train['label']

x_test_text = test['text'].fillna('')
y_test = test['label']

vectorizer = CountVectorizer()
x_train1 = vectorizer.fit_transform(x_train_text)
x_test1 = vectorizer.transform(x_test_text)

model1 = MultinomialNB()
model1.fit(x_train1,y_train)
y_pred1 = model1.predict(x_test1)

print("Multinomial Naive Bayes")
print("Accuracy: ",accuracy_score(y_test,y_pred1))
print("F1 Score: ",f1_score(y_test,y_pred1,average = 'weighted'))

binary_vectorizer = CountVectorizer(binary = True)
x_train2 = binary_vectorizer.fit_transform(x_train_text)
x_test2 = binary_vectorizer.transform(x_test_text)

model2 = BernoulliNB()
model2.fit(x_train2,y_train)
y_pred2 = model2.predict(x_test2)

print("Bernoulli Naive Bayes")
print("Accuracy: ",accuracy_score(y_test,y_pred2))
print("F1 Score: ",f1_score(y_test,y_pred2,average = 'weighted'))