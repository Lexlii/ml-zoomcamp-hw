FROM python:3.12.1

RUN pip install uv

WORKDIR /app

COPY ".python-version", "pyproject.toml", "uv.lock", ./

RUN uv sync --locked

COPY "predict.py", "model.bin",  ./

EXPOSE 9696

ENTRYPOINT [ "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]