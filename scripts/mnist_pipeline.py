import os
import torch
import torchvision
from torchvision import datasets, transforms

def download_and_preprocess_mnist(data_dir='./data'):
    BATCH_SIZE = 128

    # Ensure the data_dir is absolute
    data_dir = os.path.abspath(data_dir)

    # Transform function including normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    train_data = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Force the dataset to be processed and cached
    print(f"Number of training samples: {len(train_data)}")

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    test_data = datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform
    )

    # Force the dataset to be processed and cached
    print(f"Number of test samples: {len(test_data)}")

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, drop_last=True
    )

    print('MNIST dataset downloaded and processed successfully')
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = download_and_preprocess_mnist()
