import os
import requests
import shutil
import tarfile
import zipfile
import gzip

def download_and_extract(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    filename = url.split("/")[-1]
    file_path = os.path.join(dest_dir, filename)

    # Download the file
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    print(f"Downloaded to {file_path}")

    # Extract if compressed
    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
    elif tarfile.is_tarfile(file_path):
        with tarfile.open(file_path, 'r:*') as tar_ref:
            tar_ref.extractall(dest_dir)
    elif file_path.endswith(".gz") and not file_path.endswith(".tar.gz"):
        extracted_path = os.path.join(dest_dir, filename[:-3])
        with gzip.open(file_path, 'rb') as f_in, open(extracted_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        print("File is not compressed, nothing to extract.")
        return

    os.remove(file_path)
    print("Extraction complete and compressed file deleted.")

# Example usage
download_and_extract("https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/t93xcwm75r-11.zip", r"E:\CNR_2025\data\external")
    ######################
    # used urls 
    # https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/t93xcwm75r-11.zip