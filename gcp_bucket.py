import os
from google.cloud import storage
import pandas as pd
import numpy as np
import io
from cv2 import cv2

global bucket_name
bucket_name = "stone-resource-311918-bucket"
def bucket_file_url(file_name):
    url = "https://storage.googleapis.com/" + bucket_name + "/{}".format(file_name)
    return url

####本機上測試用####
#PATH = os.path.join(os.getcwd(), 'gcp_bucket.json')
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = PATH
#bucket = storage.Client(PATH).get_bucket(bucket_name)

####deploy時使用####
CLOUD_STORAGE_BUCKET = os.environ['CLOUD_STORAGE_BUCKET']
storage_client = storage.Client()
bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)

def get_img_idx():
    blob = bucket.get_blob("img.csv")
    blob.download_to_filename('./static/img.csv')
    img_df = pd.read_csv("./static/img.csv")
    img_idx = np.max(img_df["index"]) + 1
    img_df = img_df.append(pd.Series({'index': img_idx}), ignore_index = True)
    img_df.to_csv("./static/img.csv", index = False)
    blob = bucket.blob("img.csv")
    blob.upload_from_filename("./static/img.csv", content_type = 'text/csv')
    return img_idx

def write_png_bucket(image, file_name):
    temp_img_url = "./static/temp.png"
    cv2.imwrite(temp_img_url, image)
    blob = bucket.blob(file_name)
    blob.upload_from_filename(temp_img_url, content_type = 'image/png')
