FROM agrigorev/model-2025-hairstyle:v1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libopenblas-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch torchvision \
    onnxruntime \
    opencv-python \
    pillow \
    numpy

WORKDIR /workspace

COPY test.py /workspace/

CMD ["python3", "test.py"]
