import torch
import torchvision
import torchvision.transforms as transforms
from sklearn import preprocessing


def CIFAR10_dataset(data_path, batch, augment=False):
    """
    Retrieves CIFAR10 dataset and downloads if missing from data_path.
    Normalizes CIFAR10 dataset. Returns dataloaders for train and validation sets.
    """
    # Create a transform to normalize our CIFAR10 data from [0,255] -> [0,1] -> [-1,1]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if augment:
        # augment half of training data with horizontal flips
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.ToTensor(), 
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        # Use same transform as test set
        train_transform = transform

    # Define our train and validation sets including their loaders for CIFAR-10:
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False, num_workers=2)
    return (trainloader, testloader)

def MNIST_cluster(data_path: str):
    """
    Retrieves MNIST dataset and downloads if missing from data_path.
    Normalizes MNIST dataset (maintains 28x28 size). Returns dataloaders for train and validation sets.
    """
    # Create a transform to normalize our MNIST data:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Define our train and validation sets including their loaders for MNIST:
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    # Get indices in training set of all samples with labels 1 or 7:
    indices_0 = trainset.targets == 0
    indices_4 = trainset.targets == 4
    indices = indices_0 + indices_4
    unlabeled_data = trainset.data[indices]
    unlabeled_data = torch.flatten(unlabeled_data, 1, 2) # flatten images
    return unlabeled_data

def FashionMNIST_dataset(data_path, batch):
    """
    Retrieves FASHIONMNIST dataset and downloads if missing from data_path.
    Normalizes MNIST dataset (maintains 28x28 size). Returns dataloaders for train and validation sets.
    """
    # Create a transform to normalize our MNIST data:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # Define our train and validation sets including their loaders for MNIST:
    trainset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)
    testset = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False, num_workers=2)
    return (trainloader, testloader)


def Dataset_by_label(trainset, labels, n_samples, encode=True):
    """
    Returns samples in dataset that have label in labels.

    n_samples: # of samples to select for each label in labels

    Heavily influenced by: 
    https://github.com/Cinofix/poisoning-clustering/blob/main/experiments/run_fashion.py 
    """
    x = trainset.data
    y = trainset.targets
    label_encoder = preprocessing.LabelEncoder() # used to encode labels from: 0 to n_classes

    # Select samples in entire dataset based on labels:
    all_idxs = torch.tensor([], dtype=torch.long)
    for label in labels:
        l_mask = y == label
        label_idxs = torch.nonzero(l_mask, as_tuple=False) # get only sample indices with target == label
        selected_idxs = label_idxs[torch.randperm(len(label_idxs))][:n_samples] # Get random selection of n_samples samples
        all_idxs = torch.cat((all_idxs, selected_idxs), dim=0) # add selected samples for label to all selected
    
    # Get selected samples and transform images:
    X = x[all_idxs].view(-1, 28*28).float() # get only selected samples and flatten images
    X = X.unsqueeze(2) # makes X 3D - done to make MNIST match shapes for CIFAR and other colored images
    X /= 255.0 # normalize
    Y = y[all_idxs].view(-1)
    if encode:
        # Convert labels (e.g. 4, 7) to 0 - n_classes (e.g. 0, 1):
        Y = torch.from_numpy(label_encoder.fit_transform(Y)) 
    
    return X, Y