# specify how we want our Docker image to be created 
# define the base/ parent image we will use for this project:
FROM python:3.9-slim



# define the working directory we will have inside our docker image 
WORKDIR /ml_app

# copy everything inside our current root folder to the workdir inside Docker image 
COPY . /ml_app/

# run the command to install the needed dependencies
RUN pip install --no-cache-dir pandas scikit-learn==1.6.1




# define the default command that we will use when the container is launched
# python email_spam_detector.py - this will train our model again 
# so we can use it later for predicting spam or not spam based on 
# text added from keyboard. 
CMD [ "python" , "email_spam_detector.py"]

# the docker command we use to build this docker image will be:
    # docker built -t spam-detection . 
    # the name of the image will be spam-detection 
    # the dot(.) is used for docker to look for all the files in the current folder 


# the command for create and start a container from the existing image

    #  docker run --rm spam-detection 
# --rm - it means remove the container after it stops to save space.
# What This Command Does
# Finds the spam-detection image on your system.
# Creates a new container based on that image.
# Runs the container.
# Removes it (--rm) automatically after execution.


# now our code will run inside this docker container and not on our machine.

# if we want to run our predict.py script we can use:

    # docker run --rm spam-detection python predict.py "free mmoney now!"
# Answer we got is:
#     docker run --rm spam-detection python predict.py "free mmoney now!"
# The message is: Not Spam