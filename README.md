# Spam Detector - Machine Learning Model (Dockerized)

This project is a **Machine Learning-based Spam Detector**, which classifies messages as spam or not spam using a trained **Naive Bayes model**. The model is built using **scikit-learn** and is packaged inside a **Docker container** for easy deployment and usage.

## Features
- Trains a **Naive Bayes** model using a labeled dataset of SMS messages.
- Uses **TF-IDF vectorization** to transform text into numerical features.
- Saves the trained model and vectorizer for future predictions.
- Provides a **Dockerized environment** for easy execution.
- Allows spam detection for new messages via a command-line interface.

## Dataset
The model is trained on a publicly available **SMS spam dataset**:
[https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv)

## Project Structure
```
📂 ml_spam_detector
├── Dockerfile
├── email_spam_detector.py  # Script to train and save the model
├── predict.py              # Script to predict if a message is spam
├── spam_detector_model.pkl  # Trained model (saved)
├── spam_detector_vectorizer.pkl  # TF-IDF vectorizer (saved)
└── README.md
```

## Setup & Installation
### Running with Docker
Ensure you have **Docker** installed on your machine. Then, follow these steps:

1. **Build the Docker Image:**
   ```sh
   docker build -t spam-detection .
   ```

2. **Run the Model Training Script:**
   ```sh
   docker run --rm spam-detection
   ```

3. **Run a Prediction on a Sample Message:**
   ```sh
   docker run --rm spam-detection python predict.py "Win a free iPhone now!"
   ```

## Usage
### Training the Model
The `email_spam_detector.py` script downloads the dataset, trains the model, and saves it for future use.

To manually run it (without Docker):
```sh
python email_spam_detector.py
```

### Predicting Spam
The `predict.py` script loads the trained model and classifies a given message as **Spam** or **Not Spam**.

Example usage:
```sh
python predict.py "Congratulations! You've won a free vacation!"
```

## Dependencies
This project requires:
- Python 3.9+
- pandas
- scikit-learn 1.6.1
- joblib

If running locally, install dependencies using:
```sh
pip install pandas scikit-learn==1.6.1 joblib
```

## License
This project is open-source and available for use under the **MIT License**.

---
Happy coding! 🚀

