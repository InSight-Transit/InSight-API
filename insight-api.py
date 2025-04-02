from fastapi import FastAPI, File, UploadFile, Form
from deepface import DeepFace
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from io import BytesIO
import sys
import os
from dotenv import load_dotenv
import json
import tempfile



load_dotenv()
uri = os.getenv('URI')
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["insight"]
collection = db["facedata"]
app = FastAPI()

@app.get('/')
async def hello_world():
    result = DeepFace.verify(img1_path = "img1.jpg", img2_path = "img2.jpg")
    return {'Result' : result.verified}

@app.post('/api/addface')
async def addface(user_id : int = Form(...), file: UploadFile = File(...)):
    if file.content_type not in ['image/jpeg', 'image/png']:
        return {"error": "File type not supported. Please upload a JPEG or PNG image."}
    
    img_content = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(img_content)
        temp_file_path = temp_file.name

    embedding_objs = DeepFace.represent(
        img_path = temp_file_path, model_name="Facenet", detector_backend="mtcnn"
    )
    embedding = embedding_objs[0]["embedding"]

    collection.insert_one({"user_id" : user_id, "embedding" : embedding})

    return {'Message' : 'Face successfully added'}

@app.post('/api/deletefacedata')
async def deleteface(user_id : int = Form(...)):
    result = collection.delete_many({
        'user_id': user_id
    })

    return {'Deleted' : result.deleted_count}

@app.post('/api/search')
async def searchfaces(file : UploadFile):
    if file.content_type not in ['image/jpeg', 'image/png']:
        return {"error": "File type not supported. Please upload a JPEG or PNG image."}
    
    img_content = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(img_content)
        temp_file_path = temp_file.name

    embedding_objs = DeepFace.represent(
            img_path = temp_file_path, model_name="Facenet", detector_backend="mtcnn"
        )
    embedding = embedding_objs[0]["embedding"]

    result = client['insight']['facedata'].aggregate([
        {
            '$addFields': {
                'target_embedding': embedding
            }
        }, {
            '$unwind': {
                'path': '$embedding',
                'includeArrayIndex': 'embedding_index'
            }
        }, {
            '$unwind': {
                'path': '$target_embedding',
                'includeArrayIndex': 'target_embedding_index'
            }
        }, {
            '$project': {
                'user_id': 1,
                'embedding': 1,
                'target_embedding': 1,
                'compare': {
                    '$cmp': [
                        '$target_embedding_index', '$embedding_index'
                    ]
                }
            }
        }, {
            '$match': {
                'compare': 0
            }
        }, {
            '$group': {
                '_id': '$user_id',
                'distance': {
                    '$sum': {
                        '$pow': [
                            {
                                '$subtract': [
                                    '$embedding', '$target_embedding'
                                ]
                            }, 2
                        ]
                    }
                }
            }
        }, {
            '$project': {
                '_id': 1,
                'distance': {
                    '$sqrt': '$distance'
                }
            }
        }, {
            '$project': {
                '_id': 1,
                'distance': 1,
                'cond': {
                    '$lte': [
                        '$distance', 10
                    ]
                }
            }
        }, {
            '$match': {
                'cond': True
            }
        }, {
            '$sort': {
                'distance': 1
            }
        }
    ])
    result = list(result)

    return {'Account ID' : result[0]['_id']}
