FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

# Expose the port
EXPOSE 8002

# Run the application
CMD ["uvicorn", "website_redesign_agent:app", "--host", "0.0.0.0", "--port", "8002"]
