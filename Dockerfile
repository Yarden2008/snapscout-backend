FROM python:3.10

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port Render uses
ENV PORT=10000
EXPOSE 10000

# Start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
