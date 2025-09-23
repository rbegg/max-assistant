# 1. Use the official image from the creator of FastAPI as the base.
# This image is optimized for production with Gunicorn managing Uvicorn workers.
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

# 2. Set the working directory inside the container.
WORKDIR /app

# 3. Copy the requirements file first to leverage Docker's build cache.
# This layer will only be re-built if the requirements.txt file changes.
COPY ./requirements.txt .

# 4. Install the Python dependencies.
# --no-cache-dir ensures that pip does not store a cache, keeping the image smaller.
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 5. Copy the rest of your application's source code into the working directory.
COPY . .

# The base image provides a default command to start the server.
# It will automatically run the 'app' variable in the 'main.py' file.