try:
    from google.cloud import storage
    import google.cloud.storage
    import json
    import os
    import sys
    import pandas as pd
    import io
    from cv2 import cv2
    import tempfile
    import tensorflow as tf
except Exception as e:
    print("Error: {} ".format(e))

PATH = os.path.join(os.getcwd(), 'static/gcp-bucket.json')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = PATH
storage_client = storage.Client(PATH)
bucket = storage_client.get_bucket('ageless-aura-311408-bucket')

def write_csv_bucket(df, file_name):
    blob = bucket.blob(file_name)
    blob.upload_from_string(df.to_csv(index = False), 'text/csv')

def read_csv_bucket(file_name):
    query = io.BytesIO(bucket.blob(blob_name = file_name).download_as_string())
    df = pd.read_csv(query, index_col = None, encoding = 'UTF-8', sep = ',')
    return df

def write_png_bucket(image, file_name):
    temp_name = "temp.png"
    cv2.imwrite(temp_name, image)
    blob = bucket.blob(file_name)
    blob.upload_from_filename(temp_name, content_type = 'image/png')

def read_img_bucket():
    image = cv2.imread("https://storage.cloud.google.com/ageless-aura-311408-bucket/yaya.png")
