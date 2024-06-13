import os
import subprocess

from scripts.shakespeare_utils.utils_centralised import centralized, CustomDataset
from models.models import CharLSTM
from scripts.shakespeare_utils.utils_fedyin import ShakespeareObjectCrop

iid = True

# Install requirements
print("INSTALLING REQUIREMENTS")
subprocess.run(["pip3", "install", "-r", "/data/shakespeare_leaf/requirements.txt"])
print("LOAD DATASET")

if iid:
    subprocess.run(["/data/shakespeare_leaf/data/shakespeare/preprocess.sh", "-s", "iid", "--iu", "0.089", "--sf", "1.0", "-k", "2000", "-t", "sample", "-tf", "0.8"])

storage_path = '/data/shakespeare_leaf/data/shakespeare/data/'

if iid:
    print("iid preprocessing")
    name = 'shakespeare'
    data_obj = ShakespeareObjectCrop(storage_path, name)

# Concatenate all the clients by reshaping
reshaped_arr_x = data_obj.clnt_x.reshape(2000 * 100, 80)
reshaped_arr_y = data_obj.clnt_y.reshape(2000 * 100)

centralized_dataset_train_val = {"x": reshaped_arr_x, "y": reshaped_arr_y}

# Create test_dataset, reshaping our label array
centralized_dataset_test = {"x": data_obj.tst_x, "y": data_obj.tst_y.reshape(40000)}

train_data = CustomDataset(centralized_dataset_train_val)
# val_data = CustomDataset(centralized_dataset_train_val)
test_data = CustomDataset(centralized_dataset_test)

best_model_path = centralized("CosineAnnealingLR", train_data, test_data, test_data, learning_rates=[0.01], batch_sizes=[64])