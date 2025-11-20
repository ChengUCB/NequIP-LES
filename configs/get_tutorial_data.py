"""
Script to download tutorial data for NequIP-LES
Downloads water dataset from ChengUCB/les_fit repository
"""

from nequip.utils import download_url
import os

# Create data-sets directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(os.getcwd()), "data-sets")
os.makedirs(data_dir, exist_ok=True)

# URLs for the training and test datasets
train_url = "https://raw.githubusercontent.com/ChengUCB/les_fit/main/data-benchmark/train-H2O_RPBE-D3.xyz"
test_url = "https://raw.githubusercontent.com/ChengUCB/les_fit/main/data-benchmark/test-H2O_RPBE-D3.xyz"

# Download the files
print("Downloading training data...")
train_path = download_url(train_url, data_dir, filename="train-H2O_RPBE-D3.xyz")
print(f"Downloaded training data to {train_path}")

print("\nDownloading test data...")
test_path = download_url(test_url, data_dir, filename="test-H2O_RPBE-D3.xyz")
print(f"Downloaded test data to {test_path}")

print("\nAll data files downloaded successfully!")
