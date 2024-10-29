FROM python:3.11-slim

# Copy the current directory contents into the container at /app
COPY . /app 

# Set the working directory in the container
WORKDIR /app



# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install tensorflow flask pillow

CMD python app.py
