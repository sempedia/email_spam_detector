# we have a new script here called 'predict.py' used to predict 
# if an email is spam or not but based on a non-labeled test file 
# and using the model that we created earlier in our 'email_spam_detector.py' script.
# and that we dumped into the database along with the vectorizer 
# 'spam_detector_model.pkl' 
# 'spam_detector_vectorizer.pkl'


import joblib
import sys



# load our existing trained model from database and our vectorizer
def load_model():
    model = joblib.load('spam_detector_model.pkl')
    vectorizer = joblib.load('spam_detector_vectorizer.pkl')
    return model, vectorizer



# predict based on a new text unlabeled 
def predict(text):
    # get our model and vectorizer
    model, vectorizer = load_model()
    # use the transform() method of the vectorizer to transfrom
    # the text given into numerical values of 0 and 1 values
    text_transformed = vectorizer.transform([text])
    # we predict based on this transfromed text into numerical data
    prediction = model.predict(text_transformed)
    # return a response wheather the text is spam of not spam.
    # based on the 1 or 0 values we find in the transformed text.
    return 'Spam' if prediction[0] == 1 else 'Not Spam'


if __name__ == '__main__':
    # get the text from the keyboard . The text is added by the user
    text = sys.argv[1]
    prediction = predict(text)
    print(f'The message is: {prediction}')
    
    
    
    
# we now need to create a docker image that is like a package that contains
# everything we need to run this code on every machine we want 
# without care of the dependencies.
# for this we create a Dockerfile inside our root folder 
