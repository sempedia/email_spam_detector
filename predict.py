import joblib
import sys

def load_model():
    """
    Load the trained spam detection model and the corresponding vectorizer.
    
    Returns:
        tuple: (model, vectorizer) - The trained model and vectorizer loaded from disk.
    """
    model = joblib.load('spam_detector_model.pkl')
    vectorizer = joblib.load('spam_detector_vectorizer.pkl')
    return model, vectorizer

def predict(text):
    """
    Predict whether a given email message is spam or not.
    
    Args:
        text (str): The email content to be classified.
    
    Returns:
        str: 'Spam' if the email is classified as spam, otherwise 'Not Spam'.
    """
    model, vectorizer = load_model()
    text_transformed = vectorizer.transform([text])  # Convert text to numerical features
    prediction = model.predict(text_transformed)  # Get the model's prediction
    return 'Spam' if prediction[0] == 1 else 'Not Spam'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py '<email_text>'")
        sys.exit(1)
    
    email_text = sys.argv[1]
    result = predict(email_text)
    print(f'The message is: {result}')