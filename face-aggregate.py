from deepface import DeepFace
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import sys
import os
from dotenv import load_dotenv
import json

load_dotenv()

uri = os.getenv('URI')

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

img_path = sys.argv[1] #Put image here

embedding_objs = DeepFace.represent(
        img_path = img_path, model_name="Facenet", detector_backend="mtcnn"
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


print("ID Number")
print(result[0]['cond'])


