import os
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import densenet

app = FastAPI()
model = densenet.DenseNet()


class Image(BaseModel):
    image: str


@app.post("/predict")
async def predict(img: Image):
    if os.path.isfile(img.image):
        x = model.process_image(img.image)
        prediction = model.make_prediction(x)

        return {"response": prediction}
    else:
        raise HTTPException(status_code=404, detail="File not found")


if __name__ == '__main__':
    uvicorn.run(app)
