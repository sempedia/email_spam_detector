# give a bunch of emails to our machine learning model to can automatically detect
# the wheter that email will be spam or not spam.
# because a machine learning project will use so many libraries as sklearn, joblib, pandas
# and we want to avoid dependencies errors between those libraries and other 
# libraries we already have on ourcomputer we need to use Docker for containeraize our app.
# Docker is industry standard way of deploying the Machine learning applications.

# we train our model by using this set of sms-s , emails as test data:

# https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

# the data set is already pre-labeled - we already have labels for all those emails 
# or text messages: some of them are marked as spam and some of them are marked as ham.

# the model will learn from this test data what attributes will have a spam email or a not-spma email.

# after the model training has finished, if we give our model text 
# that is not pre-labeled , the model will predict which email will be spam or not.
# based on broad terms (high level terms).

# we will feed this test-data to a machine learning model using a library called 
# sikit-learn - that will give us a ton of tools to train our machine learning model.

# The model will look at all the words inside the tes-data file and will detect 
# which words are commonly inside a spam message 


# import a data analysis library use for importing our test-data 
# and to work with it.
# our test data is actually a csv file
import pandas as pd

# install scikit-learn module: pip install scikit-learn and import train-test-split and 
# TFidfVectorizer for working with the ML Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# to can use the same MultinomialNB model to predict based on non-labeled data 
# we can use joblib - a tool that allows us to save many different big  
# models into a database and reuse the model
import joblib

def load_data():
    """ Load the test data csv file"""
    
    # this is our csv file with test data
    url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
    
    # read the data from the csv file
    # sep='\t' means that the data is separated by tabs
    # because our labels and our text are separated by a TAB
    # header=None- means we have no header inside our file 
    # names = [] - we define names for our labesl and text 
    data = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
    
    
    # we define the data label were we are mapping these values
    # from the text bvalues into numerical values.
    # spam will be 1 
    # ham will be 0
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    # Return the data as a pandas data frame with a 0 and 1 values for spam and ham 
    return data
    
   
 
 
# train the machine leqarning model:
def train_model(data):
    # get the data from the label and text and define it as 
    # 2 new variables x and y 
    x = data['text']
    y = data['label']
    
    # use a vectorizer
    # the vectorizer is a library of skitlearn that hel us 
    # to convert a collection of raw documents into a matrix of TF-IDF features.
    # Term Freaquency-Inverse document Freaquency - measure how important is a word 
    # in a document relative to a collection of documents
    # our document is our email/sms 
    # our term freqquency - how often a word appears in our email
    # example if the word money appears 3 times in a document this means the 
    # term freaquency is 3/100 or 0,03 
    
    # Inverse document freaquency - when you have a bunch of documents (emails) like we have 
    # IDF - means how unique is a word accross all of those documents (emails.)
    # when we combine the TF and IDF we obtain how important a word is in a specific document
    # so TF and IDF gives us a numeric way of determining how important is a word
    # in our emails 
    vectorizer = TfidfVectorizer()
    # using this vectorizer will will call fit_transform that will take the x data 
    # and we will assign back this x data to variable x 
    # we use this 'vectorizer' so we will construct a matrix for x data 
    # were every row in the matrix will represent one email 
    # and every element in that row will be the TF-IDF score for a specific word
    # in that email
    x = vectorizer.fit_transform(x)
    
    # train_test_split - will split our data into a training and a testing split
    # we usually use 80% of the data to train our model 
    # and 20% of the data will be used to test the accuracy of our model
    # the parasmeter test_zize will be 0.2 in this case
    # we do the split of the data into more variables
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    # we decide which algorithm we will use to train our model
    # in this case we will use the algorithm /model called 'MultinomialNB" 
    # we set the variable 'model' equal to MultinomialNB()
    # this is a pre-built model from the sklearn library and we don't need to do anything else 
    # just assign this to our variable called model and the model will do it's work.
    # this model/algorithm - is a Naive-Bayes classifier for multinomial models.
    # this is suitable for classification with discrete features like 
    # word counts for text classification
    # the multinomial distribution usually ask for integer feature counts but it also worksw 
    # with TF-IDF fractional counts
    # this will calculate the probability to appear a given worg in an email 
    # that is not spam or that is spam type.
    # then, based on the probabilities of each word to be or not into a spam email 
    # the model will also calculate the total probability if an entire email 
    # of words can be spam or not.
    # the calculation is made based on the Bayes's Theorem:
    
    # P(A|B) = P(B|A) * P(A) / P(B)
    # This theorem assumes that the presence of each word in the email is conditionally 
    # independent of the presence of the other words in the email, given the fact that 
    # we know that the email is spam or not. 
    
    # example od data set: 
    
    # Email Text                            Label 
    # "Free money now!!!"                  Spam
    # 'Hi there, can we meet tomorrow?'    Not Spam
    # 'Earn cash fast with simple tip.'    Spam 
    
    
    # calculate the probability for each word to be spam or not spam
    # P(free|spam) = 0.5
    # P(free|not spam) = 0.4
    # P(money|spam) = 0.3
    # P(money|not spam) = 0.2
    # P(now|spam) = 0.1
    # P(now|not spam) = 0.2
    
    
    # calculate the probability that a given email/text is spam or not spam
    # P(free money now!!!|spam) = (0.5 * 0.3) / 0.5 = 0.15
    # P(free money now!!!|not spam) = (0.4 * 0.2) / 0.4 = 0.08
    
    # P(money now!!!|spam) = (0.3 * 0.15) / 0.5 = 0.045
    
    # P(spam|free money now!!!) = P(free money now!!!|spam) * P(spam) / P(free money now!!!)
    # P(not spam|free money now!!!) = P(free money now!!!|not spam) * P(not spam) / P(free money now!!!)
    
    
    # calculate the total percent of probablitity per each type of email(spam or not spamm)
    # P(spam) = 2/3
    # P(not spam) = 1/3
    
    
    # give a text and calculate the total percent of spam:
    # Seeing that 3/10 is the probability of being spam 
    # and that 2/10 is the probability of not being spam
    # Conclusion is that the text 'free money now!!!' is spam .
    # P(free money now!!!|spam) = 3/10
    # P(free money now!!!|not spam) = 2/10
    
    model = MultinomialNB()
   
    # we will train our model using our training data
    model.fit(x_train, y_train)
    
    # we get our predictions using our test data
    y_pred =model.predict(x_test)
    
    # we get our accuracy of prediction  based on our test data 
    # # and predictions we have already
    accuracy = accuracy_score(y_test, y_pred)
    
    # we get the classification report based again on the test data 
    # and tyhe predicted values 
    # by comparing the predicted values against the actual test values to see 
    # how accurate our model is in predicting whether the emails are spam or not spam
    # precision - how many of the emails predicted as spam are actually spam
    # recall - how many of the actual spam emails were correctly predicted as spam
    # f1-score - harmonic mean of precision and recall, which gives us a better measure 
    # of the model's performance when it has imbalanced classes.
    # support - how many emails in the test set this label represents.
    report  = classification_report(y_test, y_pred)
   
    # use joblib to add our model into a database called 'spam_detector_model.pkl'
    # so we can reuse it later on other test data that will be unlabeled as we had now.
    # we use the dump() method to add the model in the database 
    
    joblib.dump(model, 'spam_detector_model.pkl')
    
    # we also need to dump the vectorizer into another database created
    # specifically for the vectorizer
    # this is done by using the dump() method again.
    joblib.dump(vectorizer,'spam_detector_vectorizer.pkl')
    
    # we will return the accuracy and the classification report and the model itself
    return accuracy, report, model
    
    
    
    # we will return the accuracy and the classification report and the model itself
    return accuracy, report , model
   
    
    
    
    
    
    
    
    
    
    
    
 
# the main function that will run by default
if __name__ == '__main__':
    data = load_data()
    print(data)
    accuracy, report, model = train_model(data)
    print('Accuracy:', accuracy)
    print('Classification Report:', report)
    print('Model:', model)
     
     
     
 # this is the 'data'     
#   label                                               text
# 0         0  Go until jurong point, crazy.. Available only ...
# 1         0                      Ok lar... Joking wif u oni...
# 2         1  Free entry in 2 a wkly comp to win FA Cup fina...
# 3         0  U dun say so early hor... U c already then say...
# 4         0  Nah I don't think he goes to usf, he lives aro...
# ...     ...                                                ...
# 5567      1  This is the 2nd time we have tried 2 contact u...
# 5568      0               Will ü b going to esplanade fr home?
# 5569      0  Pity, * was in mood for that. So...any other s...
# 5570      0  The guy did some bitching but I acted like i'd...
# 5571      0                         Rofl. Its true to its name
# [5572 rows x 2 columns]

# this is the accuracy score of the model prediction
# Accuracy: 0.9533632286995516

# this is the classification report for our data 
# the predictions of the emials being spam or not 
# We have the precission score, the recall score , the f1-score and the support
# Classification Report:               precision    recall  f1-score   support

#            0       0.95      1.00      0.97       971
#            1       1.00      0.64      0.78       144

#     accuracy                           0.95      1115
#    macro avg       0.97      0.82      0.88      1115
# weighted avg       0.96      0.95      0.95      1115


# this is our used model
# Model: MultinomialNB()