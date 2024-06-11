import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import wandb
import numpy as np
from utils import CharLSTM
import torch.optim as optim

#PREPROCESSING FUNCTIONS

class Client:
    def __init__(self, client_id, subset, batch_size=32):
        self.client_id = client_id
        self.dataset = subset
        self.batch_size = batch_size
        self.train_dataloader = self.create_dataloader()

    def create_dataloader(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def train(self, model, criterion, optimizer, local_step=4):
        model.train()
        step_count = 0
        while step_count < local_step: 
            for inputs, labels in self.train_dataloader:
                labels = labels.squeeze(1)
                inputs, labels = inputs.cuda(), labels.cuda()  # Move data to CUDA
                optimizer.zero_grad()
                hidden = model.init_hidden(inputs.size(0))  # Initialize hidden state for current batch size
                outputs, hidden = model(inputs, hidden)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                step_count+=1
                if step_count >= local_step:
                    break
        return model
    


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    



#FUNCTIONS FOR MAIN CODE
    

def evaluate_model(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            
            labels = labels.squeeze(1)
            
            inputs, labels = inputs.cuda(), labels.cuda()  # Move data to CUDA
            hidden = model.init_hidden(inputs.size(0))
            outputs,_ = model(inputs,hidden)
            
            loss = criterion(outputs, labels)
            

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = test_loss / len(dataloader)
    accuracy = correct / total
    return accuracy, test_loss


# Set Model Weights
def set_weights(model, weights):
    assert len(weights) == len(list(model.parameters())), "Number of weights must match number of model parameters"
    for param, weight in zip(model.parameters(), weights):
        param.data.copy_(weight)

# Reset Environment
def reset_env(model, optimizer):
    for param in model.parameters():
        param.data.fill_(0)
    optimizer.zero_grad()



# Federated averaging function
def fedavg(lr,name,clients, global_model,test_loader, criterion, num_rounds, local_steps, fraction_clients, alpha=1.0, selection_method="dirichlet"):
    
    run_name=name
    wandb.init(project=f"federated_Shakespeare_",name=run_name)
    #inintialize net parameters
    input_size = 80  #Number of the input data , i.e length of the SENTENCE
    embedding_size = 8
    hidden_size = 256
    num_layers = 2
    output_size = 80  #number of distinct chatracters
    
    
    
    client_selection_counts = {client.client_id: 0 for client in clients}  # Initialize the tracking dictionary
    
    dirichlet_probs = np.random.dirichlet([alpha] * len(clients))
    
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")
        global_weights = [param.clone().detach() for param in global_model.parameters()]

        # Select a subset of clients based on the selection method
        if selection_method == "dirichlet":
            selected_clients = np.random.choice(clients, int(len(clients) * fraction_clients), p=dirichlet_probs)
        else:
            selected_clients = np.random.choice(clients, int(len(clients) * fraction_clients), p=None)
        

        client_weights = []
        for client in selected_clients:
            
            client_selection_counts[client.client_id] += 1  # Update the selection count

            local_model = CharLSTM(input_size, embedding_size, hidden_size, num_layers, output_size).cuda()
            set_weights(local_model, global_weights)
            optimizer = optim.SGD(local_model.parameters(), lr=lr, weight_decay=1e-4)

            local_model = client.train(local_model, criterion, optimizer, local_steps)
            client_weights.append([param.clone().detach() for param in local_model.parameters()])

        # Aggregate weights
        aggregated_weights = []
        for weights_list in zip(*client_weights):
            aggregated_weight = torch.mean(torch.stack(weights_list), dim=0)
            aggregated_weights.append(aggregated_weight)

        set_weights(global_model, aggregated_weights)
        accuracy, test_loss = evaluate_model(global_model, test_loader, criterion)
        wandb.log({"test_accuracy": accuracy * 100, "test_loss":test_loss})
        print(f"Global model accuracy after round {round_num + 1}: {accuracy * 100:.2f}%")
    
    wandb.finish()
    # Plot the frequency of client selection
    plt.figure(figsize=(10, 6))

    # Normalize the selection counts
    normalized_counts = [count / sum(client_selection_counts.values()) for count in client_selection_counts.values()]

    # Create the bar plot
    plt.bar(client_selection_counts.keys(), normalized_counts)
    plt.xlabel('Client ID')
    plt.ylabel('Relative frequency')
    plt.title(f'Clients distribution (gamma={alpha})')
    plt.savefig('client_selection_frequency.pdf') 
    plt.show()

    