# NVIDIA's TensorRT base image already ships a matched CUDA/cuDNN/TensorRT
# stack -- pin the tag to whatever driver version your deployment GPU runs.
FROM nvcr.io/nvidia/tensorrt:24.03-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
