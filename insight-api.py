from fastapi import FastAPI
from deepface import DeepFace

app = FastAPI()

@app.get('/')
def hello_world():
    result = DeepFace.verify(img1_path = "img1.jpg", img2_path = "img2.jpg")
    return {'Result' : result.verified}

@app.get('/mongo')

