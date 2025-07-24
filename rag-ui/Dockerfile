# Use a slim Python base
FROM python:3.11-slim

# Ensure warnings/errors show immediately
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your UI code
COPY . .

# Expose the port Gradio uses
EXPOSE 7860

# Default command to launch your ui.py
CMD ["python", "ui.py"]
