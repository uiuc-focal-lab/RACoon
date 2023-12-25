import torch
import torchvision
import torchvision.transforms as transforms
from src.common import Dataset, RavenMode
from auto_LiRPA.utils import get_spec_matrix

def prepare_data(dataset, train=False, batch_size=100):
    if dataset == Dataset.CIFAR10:
        transform_test = transforms.Compose([
            transforms.ToTensor(), ])

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=transform_test)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2)

        inputs, _ = next(iter(testloader))
    elif dataset == Dataset.MNIST:
        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data', train=train, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0,), (1,))
                                       ])),
            batch_size=batch_size, shuffle=False)
    else:
        raise ValueError("Unsupported Dataset")
    return testloader


def get_specification(dataset : Dataset, raven_mode : RavenMode,
                     count, nets, dataloading_seed):
    testloader = prepare_data(dataset=dataset, train=False, batch_size=count)
    images, labels = next(iter(testloader))
    if raven_mode in [RavenMode.UAP, RavenMode.ENSEMBLE]:
        constraint_matrices = get_spec_matrix(images, labels.long(), 10)
    else:
        raise ValueError(f'Output specification of {raven_mode} is not supported')
    
    return images, labels, constraint_matrices
    