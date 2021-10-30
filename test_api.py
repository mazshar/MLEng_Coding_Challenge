from fastapi.testclient import TestClient
from api import app

IMAGE_PATH = 'images/'
client = TestClient(app)


def test_incorrect_input():
    response = client.post("/predict", json='')
    assert response.status_code == 422


def test_file_not_found():
    response = client.post("/predict", json={"image": IMAGE_PATH + "random.jpg"})
    assert response.status_code == 404
    assert response.json() == {"detail": "File not found"}


def test_dog_image():
    response = client.post("/predict", json={"image": IMAGE_PATH + "dog.jpg"})

    assert response.status_code == 200
    assert response.json() == {"response": "standard poodle"}


def test_cat_image():
    response = client.post("/predict", json={"image": IMAGE_PATH + "sandie.jpg"})

    assert response.status_code == 200
    assert response.json() == {"response": "tabby, tabby cat"}
