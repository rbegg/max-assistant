# Use a slim base image
FROM python:3.11-slim-bookworm

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

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]