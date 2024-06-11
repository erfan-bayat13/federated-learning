from utils_centralised import centralized
from utils import CharLSTM 
from utils_fedyin import ShakespeareObjectCrop

iid=True

"!git clone https://github.com/FaureElia/shakespeare_leaf.git"
print("INSTALL REQUIREMENTS")
"!pip3 install -r /kaggle/working/shakespeare_leaf/requirements.txt"

print()
print("LOAD DATASET")

if iid==True:
  "!/kaggle/working/shakespeare_leaf/data/shakespeare/preprocess.sh -s iid --iu 0.089 --sf 1.0 -k 2000 -t sample -tf 0.8"


storage_path = '/kaggle/working/shakespeare_leaf/data/shakespeare/data/'

if iid:
  print("iid preprocessing")
  name = 'shakepeare'
  data_obj = ShakespeareObjectCrop(storage_path, name)


#concatenate all the clients by reshaping
reshaped_arr_x = data_obj.clnt_x.reshape(2000 * 100, 80)
reshaped_arr_y= data_obj.clnt_y.reshape(2000 * 100)




centralized_dataset_train_val={"x":reshaped_arr_x,"y":reshaped_arr_y}

#create test_dataset, reshaping our label array
centralized_dataset_test={"x":data_obj.tst_x,"y":data_obj.tst_y.reshape(40000)}


train_data = CustomDataset(centralized_dataset_train_val)
#val_data=CustomDataset(centralized_dataset_train_val)
test_data=CustomDataset(centralized_dataset_test)




best_model_path=centralized("CosineAnnealingLR",train_data,test_data,test_data,learning_rates=[0.01],batch_sizes = [64])