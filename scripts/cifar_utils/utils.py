import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
import torch.nn as nn
from collections import Counter


from models.models import FakeLeNet5

def partition_dataset(dataset, num_partitions):
    data_len = len(dataset)
    indices = list(range(data_len))
    partition_size = data_len // num_partitions
    partitions = [indices[i*partition_size:(i+1)*partition_size] for i in range(num_partitions)]
    return partitions


class Client:
    def __init__(self, client_id, train_dataset, indices, batch_size=64):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.indices = indices
        self.batch_size = batch_size
        self.train_dataloader = self.create_dataloader()

    def create_dataloader(self):
        subset = Subset(self.train_dataset, self.indices)
        dataloader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def train(self, model, criterion, optimizer, local_steps=4):
        self.train_dataloader = self.create_dataloader()  # Recreate dataloader to shuffle data

        model.train()
        step_count = 0  # Initialize step counter
        while step_count < local_steps:  # Loop until local steps are reached
            for inputs, labels in self.train_dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()  # Move data to CUDA
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                step_count += 1
                if step_count >= local_steps:  # Exit if local steps are reached
                    break
        return model

def sharding(dataset, number_of_clients, number_of_classes=100, random_seed=None):

    ''' Function that performs the sharding of the dataset given as input.
    dataset: dataset to be split;
    number_of_clients: the number of partitions we want to obtain;
    number_of_classes: (int or float) the number of classes inside each partition;
    random_seed: seed for the random operations'''

    # number_of_classes has to be a number in the range [1,100] (default value = 100 --> iid)
    # if a float is given, convert it into an integer and take the ceiling
    if isinstance(number_of_classes, float):
        number_of_classes = int(np.ceil(number_of_classes * 100))

    if number_of_classes < 1 or number_of_classes > 100:
        raise ValueError("number_of_classes should be an integer in the range [1, 100], or a float value in the range (0,1]")

    # define the amount of samples in each client
    N = len(dataset)
    indices = np.arange(N)
    n = N // number_of_clients
    remainder = N % number_of_clients

    if random_seed != None:
        np.random.seed(random_seed)

    # shuffle the dataset indices
    np.random.shuffle(indices)

    if number_of_classes == 100: # iid
        ### IID SETTING ###
        # for the iid case, we can just randomly assign to each client an equal amount of records
        clients_data = []
        start_index = 0
        for i in range(number_of_clients):
            end_index = start_index + n
            if i < remainder:
                end_index += 1
            clients_data.append([dataset[idx] for idx in indices[start_index:end_index]])
            start_index = end_index

        return [Client(client_id, train_dataset=client_data, indices=range(len(client_data))) for client_id, client_data in enumerate(clients_data)]

    else:
        ### NON-IID SETTING ###
        clients_data = []
        assigned_indices = set()  # to keep track of assigned indices
        for i in range(number_of_clients):
            if i < remainder:
                n += 1

            # Sample random classes for this client
            class_indices = np.random.choice(np.arange(100), size=number_of_classes, replace=False)

            # Create a boolean mask for samples belonging to the selected classes
            mask = np.isin(dataset.targets, class_indices)

            # Get indices of samples belonging to the selected classes and not already assigned
            filtered_indices = np.where(mask & ~np.isin(indices, list(assigned_indices)))[0]

            # Add filtered samples to client data
            clients_data.append([dataset[idx] for idx in filtered_indices[:n]])

            # Update assigned indices
            assigned_indices.update(filtered_indices[:n])

        return [Client(client_id, train_dataset=client_data, indices=range(len(client_data))) for client_id, client_data in enumerate(clients_data)]

    
 # plot clients' label distribution
def plot_label_distributions(clients_data, nc, J):
    num_clients = len(clients_data)
    num_cols = 10  # Number of columns in the grid
    num_rows = (num_clients + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 2), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten the 2D array of axes to make it easier to iterate over

    for client in clients_data:
        labels = [sample[1] for sample in client.train_dataset]  # Assuming each sample is a tuple (data, label)
        label_counts = Counter(labels)

        # Sorting the labels for consistent plotting
        labels_sorted = sorted(label_counts.keys())
        counts_sorted = [label_counts[label] for label in labels_sorted]

        axes[client.client_id].bar(labels_sorted, counts_sorted)
        axes[client.client_id].set_title(f'Client {client.client_id}', fontsize=8)
        axes[client.client_id].tick_params(axis='x', labelsize=6)
        axes[client.client_id].tick_params(axis='y', labelsize=6)


    plt.title(f"Clients' label distribution (N_C: {nc}, J:{J})")
    plt.tight_layout()
    plt.savefig(f'client_distribution_nc{nc}_J{J}.pdf') 
    plt.show()   
    
    

# Function to evaluate the global model
def evaluate_model(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()  # Move data to CUDA
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = test_loss / len(dataloader)
    accuracy = correct / total
    return accuracy, test_loss

def set_weights(model, weights):
    assert len(weights) == len(list(model.parameters())), "Number of weights must match number of model parameters"
    for param, weight in zip(model.parameters(), weights):
        param.data.copy_(weight)

def reset_env(model, optimizer):

    for param in model.parameters():
        param.data.fill_(0)  # Or whatever initialization strategy you use
    optimizer.zero_grad()


    

# Federated averaging function
def fedavg(clients, global_model, criterion, num_rounds, local_steps, fraction_clients, alpha=1.0, selection_method="dirichlet"):
    
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
        
        for client in selected_clients:
            client_selection_counts[client.client_id] += 1  # Update the selection count

        client_weights = []
        for client in selected_clients:
            local_model = FakeLeNet5().cuda()
            set_weights(local_model, global_weights)
            optimizer = optim.SGD(local_model.parameters(), lr=0.1, weight_decay=4e-3)

            local_model = client.train(local_model, criterion, optimizer, local_steps)
            client_weights.append([param.clone().detach() for param in local_model.parameters()])

        # Aggregate weights
        aggregated_weights = []
        for weights_list in zip(*client_weights):
            aggregated_weight = torch.mean(torch.stack(weights_list), dim=0)
            aggregated_weights.append(aggregated_weight)

        set_weights(global_model, aggregated_weights)
        accuracy, test_loss = evaluate_model(global_model, test_loader, criterion)
        #wandb.log({"test_accuracy": accuracy * 100, "test_loss":test_loss})
        print(f"Global model accuracy after round {round_num + 1}: {accuracy * 100:.2f}%")
    
    # Plot the frequency of client selection
    plt.figure(figsize=(10, 6))

    # Normalize the selection counts
    normalized_counts = [count / sum(client_selection_counts.values()) for count in client_selection_counts.values()]

    # Create the bar plot
    plt.bar(client_selection_counts.keys(), normalized_counts)
    plt.xlabel('Client ID')
    plt.ylabel('Relative frequency')
    if selection_method=="dirichlet":  
        plt.title(f'Clients distribution (gamma={alpha})')
    else:
        plt.title(f'Clients distribution (random selection)')
    plt.savefig('client_selection_frequency.pdf') 
    plt.show()