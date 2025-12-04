# 1. Base image
FROM python:3.9-slim

# 2. THE CRITICAL STEP: Install Linux dependencies
# Without this, the app fails.
RUN apt-get update && apt-get install -y figlet

# 3. Set up app
WORKDIR /app
COPY . .
RUN pip install flask

# 4. Run it
CMD ["python", "app.py"]