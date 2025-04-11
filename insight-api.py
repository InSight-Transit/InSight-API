from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from io import BytesIO
import tempfile
import sys
import os
from dotenv import load_dotenv
import json


load_dotenv()
uri = os.getenv('URI')
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["insight"]
collection = db["facedata"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.post('/api/addface')
async def addface(user_id : str = Form(...), file: UploadFile = File(...)):
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
async def deleteface(user_id : str = Form(...)):
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

    if len(result) > 0:
        return {'Account ID' : result[0]['_id']}
    else:
        return {'error' : 'No accounts connected to this face.'}
