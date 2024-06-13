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

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


## IID TEST RUNS ##
# Hyperparameters and configurations
K = 100
C = 0.1
alphas = [0.1, 0.5, 0.7]
base_num_rounds = 2000  # Base number of rounds for J=4
local_steps_list = [4]
#local_steps_list = [4, 8, 16]
#num_labels_list = [1, 5, 10, 50]


# Training and evaluation
for alpha in alphas:
    for J in local_steps_list:
        num_rounds = int(base_num_rounds * (4 / J))  # Scale num_rounds inversely with J

        # Create clients for non-IID sharding
        clients_iid = sharding(train_dataset, number_of_clients=100, number_of_classes=100, random_seed=0)
        
        # Initialize global model
        global_model = FakeLeNet5().cuda()
        criterion = nn.CrossEntropyLoss()


        # Reset model for IID sharding
        global_model = FakeLeNet5().cuda()

        run_name = f"[gamma: {alpha}] rounds: {num_rounds} C: {C} J:{J}"
        #wandb.init(project="final-cifar100-FL", name=run_name)

        # Train and evaluate for IID sharding
        print(f"Training with gamma={alpha}, J={J} (IID)")
        fedavg(clients_iid, global_model, criterion, num_rounds, J, C, alpha=alpha, selection_method="dirichlet")


# NON-IID TEST RUNS ##

# Hyperparameters and configurations
K = 100
C = 0.1
base_num_rounds = 2000  # Base number of rounds for J=4
local_steps_list = [4, 8, 16]
num_classes_list = [1, 5, 10, 50]


# Training and evaluation
for n_c in num_classes_list:
    for J in local_steps_list:
        num_rounds = int(base_num_rounds * (4 / J))  # Scale num_rounds inversely with J

        # Create clients for non-IID sharding
        clients_non_iid = sharding(train_dataset, number_of_clients=100, number_of_classes=n_c, random_seed=0)
        plot_label_distributions(clients_non_iid, n_c, J)
        
        
        # Initialize global model
        global_model = FakeLeNet5().cuda()
        criterion = nn.CrossEntropyLoss()
        
        run_name = f"rounds: {num_rounds} C: {C} J:{J} N_C: {n_c}"
        #wandb.init(project="noniid-cifar100-FL", name=run_name)

        # Train and evaluate for non-IID sharding
        print(f"Training with Nc={n_c}, J={J} (Non-IID)")
        fedavg(clients_non_iid, global_model, criterion, num_rounds, J, C, selection_method="random")