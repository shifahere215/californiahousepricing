FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy source files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000

CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:10000", "app:app"]

# # Use official lightweight Python image
# FROM python:3.9-slim

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# # Set work directory
# WORKDIR /app

# # Copy project files to the container
# COPY . /app

# # Install dependencies
# RUN pip install --no-cache-dir --upgrade pip \
#     && pip install --no-cache-dir -r requirements.txt

# # Expose port 5000 for Flask or for gunicorn
# EXPOSE 5000

# # Start the application with gunicorn
# CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5000", "app:app"]

# This Dockerfile sets up a Python environment with the necessary dependencies
# and runs a Gunicorn server to serve the application.
# It uses a slim version of the Python 3.9 image to keep the image size small.
# The application code is copied into the /app directory, and the working directory
# is set to /app. The requirements.txt file is used to install the necessary
# Python packages without caching to reduce the image size.
# The server is configured to listen on all interfaces at the specified PORT,
# with 4 worker processes to handle incoming requests efficiently.
# The CMD instruction specifies the command to run the Gunicorn server with
# the application module named 'app' located in the 'app.py' file.
# Make sure to set the PORT environment variable when running the container.
# To build the Docker image, use the command:
# docker build -t my-python-app .
# To run the Docker container, use the command:
# docker run -p 8000:8000 -e PORT=8000 my-python-app
# Note: Adjust the PORT environment variable as needed for your application.
# Ensure that the requirements.txt file is present in the same directory as this Dockerfile.
# The application should be structured such that the main entry point is app.py
# and the Flask app instance is named 'app'.
# This Dockerfile is suitable for deploying a Python web application using Gunicorn.
# Ensure that the Docker daemon is running and you have the necessary permissions
# to build and run Docker containers on your system.
# The image can be pushed to a container registry for deployment in cloud environments.
# For example, to push to Docker Hub, you can tag the image and use:
# docker tag my-python-app yourusername/my-python-app
# docker push yourusername/my-python-app
# This Dockerfile is designed to be simple and efficient for deploying Python web applications.
# Make sure to test the application locally before deploying it in a production environment.
# The application should be structured such that the main entry point is app.py
# and the Flask app instance is named 'app'.
# Ensure that the requirements.txt file is present in the same directory as this Dockerfile.
# The Dockerfile is designed to be simple and efficient for deploying Python web applications.
# Make sure to test the application locally before deploying it in a production environment.
# The application should be structured such that the main entry point is app.py
# and the Flask app instance is named 'app'.