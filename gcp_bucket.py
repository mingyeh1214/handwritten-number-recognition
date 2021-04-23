import os
from google.cloud import storage
import pandas as pd
import io
from cv2 import cv2

####本機上測試用####
PATH = os.path.join(os.getcwd(), 'gcp_bucket.json')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = PATH
bucket = storage.Client(PATH).get_bucket("ageless-aura-311408-bucket")

####deploy時使用####
#CLOUD_STORAGE_BUCKET = os.environ['CLOUD_STORAGE_BUCKET']
#storage_client = storage.Client()
#bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)

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