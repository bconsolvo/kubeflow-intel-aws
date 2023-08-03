import os
import boto3

s3 = boto3.client("s3")
BUCKET = "open-web-text"  # <- S3 bucket name here
PROCESSED_S3_URI = "processed"  # <- path inside S3 bucket

for split in ["train", "val"]:
    print(f"Downloading {split}.bin . . . . ")
    out_filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
    s3.download_file(BUCKET, f"{PROCESSED_S3_URI}/{split}.bin", out_filename)
