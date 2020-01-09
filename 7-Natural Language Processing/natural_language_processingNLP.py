# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t', quoting = 3) #col are separated by tab 3->ignore double quotes

# Cleaning the text 
import re #clean text efficiently 
import nltk 
nltk.download('stopwords') #remove irrelevent words that is not useful for predicting ans
from nltk.corpus import stopwords #corpus of all the review
from nltk.stem.porter import PorterStemmer #stemming -> taking the root of the word -> loved, loving, love will be consider as love
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ' , dataset['Review'][i]) #only keep letters a-z !st peram-what we want to remove in the text, we willl give what we don't want to remove '[^a-zA-Z]' 2nd peram-> replace the remove char by space 3rd pram-> where we want to apply this in here our 1st review
    review = review.lower() 
    review = review.split() #strings to list of words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # words of all the review that are not in stopwords pkg set() faster then list in python
    review = ' '.join(review) #list to string
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer # Creating sparse matrix using tokenization and create bag of words model
cv = CountVectorizer(max_features = 1500) #max_features = 1500 ->keeping 1500 most frequent words in our corpus here
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set -> most common decision tree, random forest, naive bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




