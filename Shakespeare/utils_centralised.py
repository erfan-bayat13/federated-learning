import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import wandb
import numpy as np
from utils import CharLSTM
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split


#FUNCTIONS FOR CREATING DATASET

def mysplit(dataset,percentage_val):

      x_train, x_val, y_train, y_val = train_test_split(dataset["x"], dataset["y"], test_size=percentage_val, random_state=42)

      train_dict = {"x": x_train,"y": y_train}

      val_dict = {"x": x_val,"y": y_val}

      return train_dict, val_dict

class CustomDataset(Dataset):
    def __init__(self, data):
        self.x = data['x']
        self.y = data['y']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    


#MAIN FUNCTIONS


def get_scheduler(optimizer, scheduler_name, num_epochs):
    scheduler_dict = {
        'StepLR': optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
        'MultiStepLR': optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1),
        'ExponentialLR': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1),
        'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs),
        'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10),
        'OneCycleLR': optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=num_epochs),
    }

    if scheduler_name in scheduler_dict:
        return scheduler_dict[scheduler_name]
    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")
    




def evaluate(lr, batch_size,model, data_loader, criterion, val=False):
    with torch.no_grad():
      model.eval()
      total_loss=0
      total_correct=0
      total_samples = 0

      for input_data, target_data in data_loader:
                input_data,target_data=input_data.cuda(),target_data.cuda()

                #initialize the hidden state as none
                hidden = None
                #predict
                output, hidden = model(input_data.squeeze(0), hidden)



                #compute loss
                loss = criterion(output, target_data)

                total_loss += loss.item()

                _, predicted = torch.max(output, 1)  # Get the index of the max log-probability



                indices_of_ones = (target_data == 1).nonzero()


                total_correct += predicted.eq(target_data).sum().item()
                total_samples += target_data.size(0)

      accuracy = total_correct / total_samples

      if val==True:
        print(f'Validation--> Loss: {total_loss/total_samples}, Accuracy: {accuracy}')
        print("_____________________________________________________________________")

        return accuracy,total_loss/total_samples

      print(f'TEST--> Loss: {total_loss/total_samples}, Accuracy: {accuracy}')
      return accuracy,total_loss/total_samples


def centralized(scheduler_name,train_data,val_data,test_data,learning_rates,batch_sizes,num_epochs=150,momentum=0.9, weight_decay=4e-4):
    best_accuracy = 0.0
    wandb.login()
    for lr in learning_rates:
        for batch_size in batch_sizes:
            
                # Define unique name for this hyperparameter configuration
                # run_name = f"Scheduler: {scheduler_name} Learning rate: {lr} Batch size: {batch_size}"
                run_name = f"Learning rate: {lr} Batch size: {batch_size}, NO DROPOUT,test2"

                # Initialize the run with unique name
                # MODIFY PROJECT NAME IT IF YOU WANT TO RUN THE CODE!!
                wandb.init(project="Centralized_Shakespeare_optimal_test_dropout", name=run_name)


                # Create DataLoader
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

                val_accuracies=[]
                train_accuracies=[]

                print("NEW CONFIGURATION")
                print("_________________________________________")
                print("_________________________________________")

                #define network paramters

                input_size = 80  #Number of the input data , i.e length of the SENTENCE
                embedding_size = 8
                hidden_size = 256
                num_layers = 2
                output_size = 80  #number of distinct chatracters
                
                model = CharLSTM(input_size, embedding_size, hidden_size,num_layers, output_size).cuda()
                # Define loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=4e-4)
                
                scheduler = get_scheduler(optimizer, scheduler_name, num_epochs)


                for epoch in range(num_epochs):
                      model.train()
                      epoch_loss = 0
                      total_correct = 0
                      total_samples = 0


                      for input_data,target_data in (train_loader):
                                input_data,target_data=input_data.cuda(),target_data.cuda()

                                optimizer.zero_grad()


                                #initialize the hidden state as none
                                hidden = None
                                #predict
                                output, hidden = model(input_data.squeeze(0), hidden)


                                #compute loss

                                loss = criterion(output, target_data)


                                # Backward pass
                                loss.backward()
                                optimizer.step()


                                epoch_loss += loss.item()

                                values, predicted = torch.max(output, 1)  # Get the index of the max log-probability

                                total_correct += predicted.eq(target_data).sum().item()
                                total_samples += target_data.size(0)






                      accuracy = total_correct / total_samples
                      scheduler.step()
                        
                      train_accuracies.append(accuracy)


                      print(f'lr: {lr}, batch_size:{batch_size}| Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/total_samples}, Accuracy: {accuracy}')
                      #validation
                      val_accuracy,val_loss=evaluate(lr,batch_size,model,val_loader,criterion,True)
                        
                      val_accuracies.append(val_accuracy)
                        


                      # Log metrics to wandb after each epoch

                      wandb.log({
                          "train_loss": epoch_loss/total_samples,
                          "train_accuracy": accuracy,
                          "val_loss": val_loss,
                          "val_accuracy": val_accuracy
                          })
                    
                #TEST THE FINAL MODEL
                    
                test_accuracy,test_loss=evaluate(lr,batch_size,model,test_loader,criterion,False)
                    
                # Save the model if it's the best so far
                if test_accuracy > best_accuracy:
                        best_accuracy = test_accuracy
                        best_model_path = f'best_model_lr{lr}_bs{batch_size}.pth'
                        torch.save(model.state_dict(), best_model_path)

                # Log the best validation accuracy and the path to the best model
                wandb.run.summary["best_val_accuracy_shakespeare"] = best_accuracy
                wandb.run.summary["best_model_path_shakespeare"] = best_model_path

                # Close the WandB run
                wandb.finish()

    print(f'Best model saved at: {best_model_path}')
    
    #plots
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1,  num_epochs+ 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs+ 1), val_accuracies, label='Validation Accuracy')

    for y in [0.25, 0.3,0.35, 0.4,0.45,0.5,0.55]:
        plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(False)
    plt.show()

    
   
    
    return best_model_path