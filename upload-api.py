from deepface import DeepFace
import json
from tqdm import tqdm
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import sys
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv('URI')

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

db = client["insight"]
collection = db["facedata"]

img_path = sys.argv[2] #Put image here

embedding_objs = DeepFace.represent(
        img_path = img_path, model_name="Facenet", detector_backend="mtcnn"
    )
embedding = embedding_objs[0]["embedding"]

collection.insert_one({"user_id" : int(sys.argv[1]), "embedding" : embedding})
