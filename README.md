# ML Engineering Coding Challenge

This is a Python project designed for Python version >=3.6.

## Setup
The following commands install the necessary tools:
```bash
pip install -r requirements.txt
```

## Running API server
To run the FastAPI server:
```bash
python api.py
```

You can now use curl to send requests:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "image": "images/dog.jpg"
}'
```

You should get a response with response code 200 and 'standard poodle' message.

## Running unit tests

To run the tests:
```bash
pytest
```