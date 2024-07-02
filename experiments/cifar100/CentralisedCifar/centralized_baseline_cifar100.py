import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# Define transformations for training and testing datasets
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),          # Randomly flip the image horizontally
    transforms.RandomCrop(32, padding=4),       # Randomly crop the image with padding
    transforms.ToTensor(),                      # Convert image to PyTorch tensor
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))  # Normalize pixel values
])

transform = transforms.Compose([
    transforms.ToTensor(),                      # Convert image to PyTorch tensor
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))  # Normalize pixel values
])

# Load the CIFAR-100 training+validation dataset
cifar100_trainval = torchvision.datasets.CIFAR100(root='./data', train=True,
                                               download=True, transform=None)

# Load the CIFAR-100 test dataset
cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False,
                                              download=True, transform = transform)

# # Calculate mean and standard deviation
# mean = np.zeros(3)
# std = np.zeros(3)

# for image, _ in cifar100_train:
#     image = np.array(image) / 255.  # Convert PIL Image to numpy array and normalize to [0, 1]
#     mean += np.mean(image, axis=(0, 1))
#     std += np.std(image, axis=(0, 1))

# # Calculate the mean and standard deviation for each channel
# mean /= len(cifar100_train)
# std /= len(cifar100_train)

# print("Mean: ", mean)
# print("Std: ", std)

# Create the validation split (Experimentation.preliminaries in the project pdf)

# Define the size of the validation set (e.g., 20% of the full dataset)
val_size = int(0.2 * len(cifar100_trainval))
train_size = len(cifar100_trainval) - val_size

# Set seed for reproducibility
generator = torch.Generator()
generator.manual_seed(42)

# Split the dataset into training and validation sets
cifar100_train, cifar100_val = torch.utils.data.random_split(cifar100_trainval, [train_size, val_size], generator=generator)

# Print the number of samples in each split
print("Training set size:", len(cifar100_train))
print("Validation set size:", len(cifar100_val))
print("Test set size:", len(cifar100_test))

# Now we can apply data augmentation to the training set only
cifar100_train.dataset.transform = transform_train
cifar100_val.dataset.transform = transform

# NOTE: we should not apply data augmentation to the validation split, that is used to evaluate the model performance

sample, _ = cifar100_train[0]
print(type(sample))

# Define a function to denormalize the image
# def denormalize(image):
#     image = image.to('cpu').numpy().transpose((1, 2, 0))
#     mean = np.array([0.5071, 0.4866, 0.4409])
#     std = np.array([0.2673, 0.2564, 0.2762])
#     image = image * std + mean
#     image = np.clip(image, 0, 1)
#     return image


# Create data loaders
trainval_loader = torch.utils.data.DataLoader(cifar100_trainval, batch_size=64, shuffle=True)
train_loader = torch.utils.data.DataLoader(cifar100_train, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(cifar100_val, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(cifar100_test, batch_size=64, shuffle=False)

# Define the custom neural network
class FakeLeNet5(nn.Module):
    def __init__(self): # initialize tha layers that you are going to use
        super(FakeLeNet5, self).__init__()
        self.flatten = nn.Flatten()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 100) # 100 is the number of classes

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 5 * 5)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.log_softmax(x, dim = 1) # dim is the direction along which softmax is computed (row-wise)

        return x

#pip install wandb

#import wandb

#wandb.login()

def param_tuning(model,learning_rates,batch_sizes,num_epochs=10,momentum=0.9, weight_decay=4e-4):
    # lr is learning rate
    # scheduler_name for lr should be one of StepLR , MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
    # lr and batchsize should be a list
    # Loop over hyperparameters
    best_accuracy = 0.0  # Variable to keep track of the best validation accuracy
    best_model_path = '/content/drive/My Drive/'
    for lr in learning_rates:
        for batch_size in batch_sizes:

            # Define unique name for this hyperparameter configuration
            # run_name = f"Scheduler: {scheduler_name} Learning rate: {lr} Batch size: {batch_size}"
            #run_name = f"Learning rate: {lr} Batch size: {batch_size} ReLU + DA"

            # Initialize the run with unique name
            # wandb.init(project="fakelenet5_hyperparam_final", name=run_name)

            # Data loaders
            train_loader = DataLoader(cifar100_train, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(cifar100_val, batch_size=batch_size, shuffle=False)

            # Initialize the network
            net = model().cuda()

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=4e-4)

            # Train the network
            for epoch in range(1, num_epochs+1):

                ####### TRAINING LOOP #######
                net.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.cuda(), targets.cuda()

                    # compute prediction and loss
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    # backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    #scheduler.step()

                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                train_loss = running_loss / len(train_loader)
                train_accuracy = 100. * correct / total
                print(f'(Learning rate:{lr}, Batch size: {batch_size}) [{epoch}/{num_epochs}] TRAINING Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

                # Save checkpoint
                torch.save(net.state_dict(), f'fakelenet5_lr{lr}_bs{batch_size}_epoch{epoch}.pth')


                ####### VALIDATION LOOP #######
                net.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(val_loader):
                        inputs, targets = inputs.cuda(), targets.cuda()
                        outputs = net(inputs)
                        loss = criterion(outputs, targets)

                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()

                # Evaluate on the validation set and print results
                val_loss = val_loss / len(val_loader)
                val_accuracy = 100. * val_correct / val_total
                print(f'(Learning rate:{lr}, Batch size: {batch_size}) [{epoch}/{num_epochs}] VALIDATION Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
                print('_'*60)

                # Log metrics to wandb after each epoch
                #wandb.log({
                      #"train_loss": train_loss,
                      #"train_accuracy": train_accuracy,
                      #"val_loss": val_loss,
                      #"val_accuracy": val_accuracy
                      #})

            # Save the model if it's the best so far
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_path = f'best_model_lr{lr}_bs{batch_size}.pth'
                torch.save(net.state_dict(), best_model_path)

            # Log the best validation accuracy and the path to the best model
            #wandb.run.summary["best_val_accuracy"] = best_accuracy
            #wandb.run.summary["best_model_path"] = best_model_path

            # Close the WandB run
            #wandb.finish()

    print(f'Best model saved at: {best_model_path}')
    return best_model_path

param_tuning(model=FakeLeNet5, learning_rates=[0.01, 0.05, 0.001], batch_sizes = [64], num_epochs=150)

"""# TRAINING THE NETWORK
With the best hyperparameters configuration
"""

# First let us apply the transformations to the FULL dataset (train+val, recall that we uploaded it with transfom=None)
cifar100_trainval.transform = transform_train

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

"""https://discuss.pytorch.org/t/scheduler-step-after-each-epoch-or-after-each-minibatch/111249
at this link it is written that the torch suggestion regarding the step of the scheduler is to put it at the end of each epoch and not after each mini batch (otherwise the lr will decay too fast), so i modified the code accordingly

"""

# Initialize the model
model = FakeLeNet5().cuda()

# Best configuration parameters
batch_size = 64
lr = 0.01

# Number of epochs
# NOTE: the number of epochs can be inferred looking at the plots in wandb.
num_epochs = 133

# Define unique name for this hyperparameter configuration
# run_name = f"Scheduler: {scheduler_name} Learning rate: {lr} Batch size: {batch_size}"
#run_name = f"Learning rate: {lr} Batch size: {batch_size} ReLU + data aug"

# Initialize the run with unique name
#wandb.init(project="fakelenet5_MODELTRAINING_final", name=run_name)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=4e-4)
scheduler_name = 'CosineAnnealingLR'
scheduler = get_scheduler(optimizer, scheduler_name, num_epochs)

# Create data loaders
trainval_loader = DataLoader(cifar100_trainval, batch_size=batch_size, shuffle=True) # using both train and validation datasets
test_loader = DataLoader(cifar100_test, batch_size=batch_size, shuffle=False)

# Training loop --> we want to iterate over the train_loader
def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader): # batch size is 64
        inputs, targets = inputs.cuda(), targets.cuda()

        # compute prediction and loss
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    scheduler.step() # scheduler step after each epoch

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'[{epoch}/{num_epochs}] TRAIN Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    return train_loss, train_accuracy


# Test loop --> we are computing the loss to evaluate the dataa
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = test_loss / len(test_loader)
    test_accuracy = 100. * correct / total
    print(f'TEST Loss: {test_loss:.6f} Acc: {test_accuracy:.2f}%')
    print('_'*60)
    return test_loss, test_accuracy

# Run the training and testing
for epoch in range(1, num_epochs + 1):
    train_loss, train_accuracy = train(epoch, model, trainval_loader, criterion, optimizer)
    test_loss, test_accuracy = test(model, test_loader, criterion)
    # wandb.log({
    #   "train_loss": train_loss,
    #   "train_accuracy": train_accuracy,
    #   "test_loss": test_loss,
    #   "test_accuracy": test_accuracy
    #   })

# wandb.finish()
