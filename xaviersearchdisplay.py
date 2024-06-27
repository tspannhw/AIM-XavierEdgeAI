from pymilvus import MilvusClient
import numpy as np
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from pymilvus import connections
from pymilvus import utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
import uuid
from time import sleep
from math import isnan
import time
import sys
import datetime
import subprocess
import sys
import os
import traceback
import math
import base64
import json
from time import gmtime, strftime
import random, string
import base64
import socket 
import glob
import torch
from torchvision import transforms
from PIL import Image
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import multiprocessing
import cv2
import time
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import io

# -----------------------------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------------------------

DIMENSION = 512

DATABASE_NAME = "./XavierEdgeAI.db"

COLLECTION_NAME = "XavierEdgeAI"

PATH = "/home/nvidia/nvme/images/"

slack_token = os.environ["SLACK_BOT_TOKEN"]

BLIP_MODEL = "Salesforce/blip-image-captioning-large"

# -----------------------------------------------------------------------------------------------
# Slack
# -----------------------------------------------------------------------------------------------

client = WebClient(token=slack_token)

# -----------------------------------------------------------------------------------------------
# Milvus Feature Extractor
# -----------------------------------------------------------------------------------------------

class FeatureExtractor:
    def __init__(self, modelname):
        # Load the pre-trained model
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.model.eval()

        # Get the input size required by the model
        self.input_size = self.model.default_cfg["input_size"]

        config = resolve_data_config({}, model=modelname)
        # Get the preprocessing function provided by TIMM for the model
        self.preprocess = create_transform(**config)

    def __call__(self, imagepath):
        # Preprocess the input image
        input_image = Image.open(imagepath).convert("RGB")  # Convert to RGB if needed
        input_image = self.preprocess(input_image)

        # Convert the image to a PyTorch tensor and add a batch dimension
        input_tensor = input_image.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Extract the feature vector
        feature_vector = output.squeeze().numpy()

        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()

extractor = FeatureExtractor("resnet34")

# -----------------------------------------------------------------------------------------------
# Milvus Collection
# -----------------------------------------------------------------------------------------------

milvus_client = MilvusClient(DATABASE_NAME)

# -----------------------------------------------------------------------------------------------
# OpenCV From Webcam
# -----------------------------------------------------------------------------------------------

cam = cv2.VideoCapture(0)
result, image = cam.read()
query_image = PATH + 'xaviernow.jpg'

if result:
    cv2.imwrite(query_image, image)
else:
    print("No image")

# -----------------------------------------------------------------------------------------------
# Metadata Fields
# -----------------------------------------------------------------------------------------------

currenttimeofsave = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
hostname = os.uname()[1]

print(hostname)
print(currenttimeofsave)

# -----------------------------------------------------------------------------
# Image Search
# -----------------------------------------------------------------------------

try:
    results = milvus_client.search(
        COLLECTION_NAME,
        data=[extractor(query_image)],
        output_fields=["filename", "caption", "currenttime", "id"],
        search_params={"metric_type": "COSINE"},
    )

    mfilename = ""
    mfilepath = ""
    mcaption = ""
    mid = 0
    mcurrenttime = ""
    mcount = 0
    mdistance = 0

    for result in results:
        # print(result)
        mcount = mcount + 1

        for hit in result[:10]:
            mfilename = hit["entity"]["filename"]
            mcaption = hit["entity"]["caption"]
            mid = hit["id"]
            mdistance = hit["distance"]
            mcurrenttime = hit["entity"]["currenttime"]
 
            img_color = cv2.imread(mfilename,cv2.IMREAD_COLOR)
            cv2.imshow('Milvus Result ' + str(mcaption),img_color)
            cv2.waitKey(0)
            
            print(str(mfilename) + " with caption: " + str(mcaption))

            try:
                response = client.chat_postMessage(
                    channel="C06NE1FU6SE",
                    text=(f"NVIDIA Xavier ({hostname}) Milvus Search Result\nDistance: {mdistance}\n{mfilename}\n{mcaption} @ {mcurrenttime} \n ")
                )
            except SlackApiError as e:
                # You will get a SlackApiError if "ok" is False
                assert e.response["error"]

            try:
                response = client.files_upload_v2(
                    channel="C06NE1FU6SE",
                    file=mfilename,
                    title="NVIDIA Xavier Milvus Search Result",
                    initial_comment="Milvu Image Search Result",
                )
            except SlackApiError as e:
                assert e.response["error"]

except Exception as e:
    print("An error:", e)

cv2.destroyAllWindows()
print("Search done")
