# Use a lightweight Python 3.9 image as the base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /ml_app

# Copy all project files into the container
COPY . /ml_app/

# Install required dependencies
RUN pip install --no-cache-dir pandas scikit-learn==1.6.1

# Set the default command to run the spam detection script
CMD ["python", "email_spam_detector.py"]

# Instructions for building and running the Docker container:
# Build the Docker image: docker build -t spam-detection .
# Run a container from the image: docker run --rm spam-detection
# Run the predict script inside the container:
# docker run --rm spam-detection python predict.py "free money now!"
