import os
import subprocess
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import wandb

from scripts.shakespeare_utils.utils import CharLSTM
from scripts.shakespeare_utils.utils_federated import fedavg, CustomDataset, Client
from scripts.shakespeare_utils.utils_fedyin import ShakespeareObjectCrop, ShakespeareObjectCrop_noniid

batch_size = 64
iid = False

# Install requirements
print("INSTALLING REQUIREMENTS")
subprocess.run(["pip3", "install", "-r", "/data/shakespeare_leaf/requirements.txt"])
print("LOAD DATASET")

if iid:
    subprocess.run(["/data/shakespeare_leaf/data/shakespeare/preprocess.sh", "-s", "iid", "--iu", "0.089", "--sf", "1.0", "-k", "2000", "-t", "sample", "-tf", "0.8"])
else:
    subprocess.run(["/data/shakespeare_leaf/data/shakespeare/preprocess.sh", "-s", "niid", "--sf", "1.0", "-k", "2000", "-t", "sample", "-tf", "0.8"])

storage_path = '/data/shakespeare_leaf/data/shakespeare/data/'

if iid:
    print("iid preprocessing")
    name = 'shakespeare'
    data_obj = ShakespeareObjectCrop(storage_path, name)
else:
    print("niid preprocessing")
    name = 'shakespeare_nonIID'
    number_of_clients = 100
    data_obj = ShakespeareObjectCrop_noniid(storage_path, name, number_of_clients)

clients = []

for client_id,(client_x,client_y) in enumerate(zip(data_obj.clnt_x,data_obj.clnt_y)):

  #create customDataset for storing client data
  client_dataset = CustomDataset(client_x, client_y)

  #create client giving as input the local dataset(no indices needed)
  client = Client(client_id=client_id, subset=client_dataset, batch_size=batch_size)
  clients.append(client)



#define a test loader for test dataset

test_dataset=CustomDataset(data_obj.tst_x, data_obj.tst_y)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)







num_rounds = 200
fraction_of_clients = 0.1
local_steps=4
criterion = nn.CrossEntropyLoss()
lr=1


#inintialize net parameters
input_size = 80  #Number of the input data , i.e length of the SENTENCE
embedding_size = 8
hidden_size = 256
num_layers = 2
output_size = 80  #number of distinct chatracters


#global_model = CharLSTM(input_size, embedding_size, hidden_size, num_layers, output_size).cuda()

local_steps_values=[4,8,16]

wandb.login()

for ls in local_steps_values:
    global_model = CharLSTM(input_size, embedding_size, hidden_size, num_layers, output_size).cuda()
    name=f"lr_{lr},ls_{ls}_niid"
    fedavg(lr,name,clients, global_model,test_loader, criterion, num_rounds, local_steps, fraction_of_clients, alpha=1.0, selection_method="random")