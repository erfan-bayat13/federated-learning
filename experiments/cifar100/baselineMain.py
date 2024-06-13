from scripts.cifar_utils.utils import *


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),          # Randomly flip the image horizontally
    transforms.RandomCrop(32, padding=4),       # Randomly crop the image with padding
    transforms.ToTensor(),                      # Convert image to PyTorch tensor
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))  # Normalize pixel values
])


# Load CIFAR-100 dataset

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

## BASELINE ##
# Hyperparameters and configurations
K = 100
C = 0.1
alphas = [1, 0.1, 0.5, 0.7]
num_rounds = 2000  # Base number of rounds for J=4
J = 4


# Training and evaluation --> BASELINE

clients_iid = sharding(train_dataset, number_of_clients=100, number_of_classes=100, random_seed=0)
# Initialize global model
global_model = FakeLeNet5().cuda()
criterion = nn.CrossEntropyLoss()
#run_name = f"[random selection] rounds: {num_rounds} C: {C} J:{J}"
#wandb.init(project="final-cifar100-FL", name=run_name)
# Train and evaluate for IID sharding
print(f"Training with J={J} (IID)")
fedavg(clients_iid, global_model, criterion, num_rounds, J, C, selection_method="random")