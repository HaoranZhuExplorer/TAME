import copy
import numpy as np
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import ConcatDataset, Dataset, TensorDataset, DataLoader
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler
import math
from collections import deque
import heapq
import random

from models import *
from prune_networks import prune_network
from prune import iterative_pruning_finetuning
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

T_MAX = 200
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
MILESTONES = [60, 120, 160, 230]
GAMMA = 0.2

# MILESTONES = [150, 225] # TODO
# GAMMA = 0.1

AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST,
    'cifar10': datasets.CIFAR10,
    'cifar100': datasets.CIFAR100,
}

AVAILABLE_TRANSFORMS_TRAIN = {
    'mnist': [
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ],
    'mnist28': [
        transforms.ToTensor(),
    ],
    'cifar10': [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ], 
    'cifar100': [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], 
                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    ],
}

AVAILABLE_TRANSFORMS_TEST = {
    'mnist': [
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ],
    'mnist28': [
        transforms.ToTensor(),
    ],
    'cifar10': [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ], 
    'cifar100': [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], 
                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    ],
}

AVAILABLE_TRANSFORMS_BUFFER_TRAIN = {
    'mnist': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'mnist28': [
        transforms.ToTensor(),
    ],
    'cifar10': [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ], 
    'cifar100': [
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], 
                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    ],
}

AVAILABLE_TRANSFORMS_BUFFER_TEST = {
    'mnist': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'mnist28': [
        transforms.ToTensor(),
    ],
    'cifar10': [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ], 
    'cifar100': [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], 
                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    ],
}

DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist28': {'size': 28, 'channels': 1, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100},
}


CIFAR_LABELS = {
0: [72, 4, 95, 30, 55],
1: [73, 32, 67, 91, 1],
2: [92, 70, 82, 54, 62],
3: [16, 61, 9, 10, 28],
4: [51, 0, 53, 57, 83],
5: [40, 39, 22, 87, 86],
6: [20, 25, 94, 84, 5],
7: [14, 24, 6, 7, 18],
8: [43, 97, 42, 3, 88],
9: [37, 17, 76, 12, 68],
10: [49, 33, 71, 23, 60],
11: [15, 21, 19, 31, 38],
12: [75, 63, 66, 64, 34],
13: [77, 26, 45, 99, 79],
14: [11, 2, 35, 46, 98],
15: [29, 93, 27, 78, 44],
16: [65, 50, 74, 36, 80],
17: [56, 52, 47, 59, 96],
18: [8, 58, 90, 13, 48],
19: [81, 69, 41, 89, 85]}


XSMALL_SIZE = 22
SMALL_SIZE = 24
MEDIUM_SIZE = 30
LARGE_SIZE = 32
plt.rc('font', size=SMALL_SIZE)          
plt.rc('axes', titlesize=MEDIUM_SIZE)     
plt.rc('axes', labelsize=MEDIUM_SIZE)    
plt.rc('xtick', labelsize=XSMALL_SIZE)    
plt.rc('ytick', labelsize=XSMALL_SIZE)    
plt.rc('legend', fontsize=XSMALL_SIZE)    
plt.rc('figure', titlesize=LARGE_SIZE)  
plt.rcParams["figure.figsize"] = (15,10)


def create_target_transform_dict(super_labels):
    target_dict = {}
    for key in super_labels:
        for idx, label in enumerate(super_labels[key]):
            target_dict[label] = idx
    return target_dict


def _permutate_image_pixels(image, permutation):
    '''Permutate the pixels of an image according to [permutation].
    [image]         3D-tensor containing the image
    [permutation]   <ndarray> of pixel-indeces in their new order'''
    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation]  #--> same permutation for each channel
        image = image.view(c, h, w)
        return image
    

def get_dataset_buffer(name, type='train', download=True, capacity=None, permutation=None, dir='./datasets',
                verbose=False, target_transform=None):
    '''Create [train|valid|test]-dataset.'''
    data_name = 'mnist' if name =='mnist28' else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    if type == 'test':
        dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS_BUFFER_TEST[name],
            transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
        ])
    else:
        dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS_BUFFER_TRAIN[name],
            transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
        ])

    # load data-set
    dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    # print information about dataset on the screen
    if verbose:
        print("  --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset = ConcatDataset([copy.deepcopy(dataset) for _ in range(int(np.ceil(capacity / len(dataset))))])
    return dataset


def get_dataset(name, type='train', download=True, capacity=None, permutation=None, dir='./datasets',
                verbose=False, target_transform=None):
    '''Create [train|valid|test]-dataset.'''
    data_name = 'mnist' if name =='mnist28' else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    if type == 'test':
        dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS_TEST[name],
            transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
        ])
    else:
        dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS_TRAIN[name],
            transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
        ])
    

    # load data-set
    dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    # print information about dataset on the screen
    if verbose:
        print("  --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset = ConcatDataset([copy.deepcopy(dataset) for _ in range(int(np.ceil(capacity / len(dataset))))])
    return dataset


class CustomBufferDataset(Dataset):
    def __init__(self, original_dataset, sub_indices, targets):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = sub_indices
        self.targets = targets
    def __len__(self):
        return len(self.sub_indeces)
    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        x = sample[0]
        y = self.targets[index]
        return (x, y)


class CustomTestDataset(Dataset):
    def __init__(self, selector_dataset, test_dataset, sub_indices, targets, task_ids, class_ids):
        super().__init__()
        self.selector_dataset = selector_dataset
        self.test_dataset = test_dataset
        self.sub_indeces = sub_indices
        self.targets = targets
        self.task_ids = task_ids
        self.class_ids = class_ids
    def __len__(self):
        return len(self.sub_indeces)
    def __getitem__(self, index):
        selector_sample = self.selector_dataset[self.sub_indeces[index]]
        selector_data = selector_sample[0]
        test_sample = self.test_dataset[self.sub_indeces[index]]
        test_data = test_sample[0]
        target = self.targets[index]
        task_id = self.task_ids[index]
        class_id = self.class_ids[index]
        return (selector_data, test_data, target, task_id, class_id)


class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].
    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''
    def __init__(self, original_dataset, sub_labels, idx, num_classes_per_task, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        self.task_id = idx
        self.num_classes_per_task = num_classes_per_task
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "train_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.train_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.train_labels[index])
            elif hasattr(self.dataset, "test_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.test_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.test_labels[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        # self.sub_indeces = self.sub_indeces[:11136]  # TODO
        self.target_transform = target_transform
    def __len__(self):
        return len(self.sub_indeces)
    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target, self.task_id, sample[1], self.sub_indeces[index])
        return sample


class SubDatasetPerm(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].
    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''
    def __init__(self, original_dataset, task_id, num_classes_per_task):
        super().__init__()
        self.dataset = original_dataset
        self.task_id = task_id
        self.num_classes_per_task = num_classes_per_task
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        sample = self.dataset[index]
        sample = (sample[0], sample[1], self.task_id, sample[1] + self.task_id * self.num_classes_per_task, index)
        return sample
    

class ExemplarDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).
    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''
    def __init__(self, exemplar_sets, target_transform=None):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.target_transform = target_transform
    def __len__(self):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            total += len(self.exemplar_sets[class_id])
        return total
    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            exemplars_in_this_class = len(self.exemplar_sets[class_id])
            if index < (total + exemplars_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                exemplar_id = index - total
                break
            else:
                total += exemplars_in_this_class
        image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
        return (image, class_id_to_return)


def get_multitask_experiment(name, scenario, tasks, new_nb_tasks, data_dir="./datasets", only_config=False, verbose=False, exception=False, super_class=False):
    '''Load, organize and return train- and test-dataset for requested experiment.
    [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                            are not shuffled before being distributed over the tasks (splitMNIST)'''

    permutations = None
    # depending on experiment, get and organize the datasets
    if name == 'permMNIST':
        # configurations
        config = DATASET_CONFIGS['mnist']
        classes_per_task = 10
        if not only_config:
            # generate permutations
            if exception:
                permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(tasks-1)]
            else:
                permutations = [np.random.permutation(config['size']**2) for _ in range(tasks)]
            # prepare datasets
            train_datasets = []
            test_datasets = []
            for task_id, p in enumerate(permutations):
                target_transform = transforms.Lambda(
                    lambda y, x=task_id: y + x * classes_per_task
                ) if scenario in ('task', 'class') else None
                mnist_train = get_dataset('mnist', type="train", permutation=p, dir=data_dir,
                                          target_transform=target_transform, verbose=verbose)
                mnist_test = get_dataset('mnist', type="test", permutation=p, dir=data_dir,
                                         target_transform=target_transform, verbose=verbose)
                print("SubDatasetPerm = " + str(SubDatasetPerm(mnist_train, task_id, classes_per_task)))
                train_datasets.append(SubDatasetPerm(mnist_train, task_id, classes_per_task))
                test_datasets.append(SubDatasetPerm(mnist_test, task_id, classes_per_task))
    elif (name == 'splitMNIST' or name == 'cifar10' or name == 'cifar100'):
        # check for number of tasks
        if ((name == 'splitMNIST' or name == 'cifar10') and tasks > 10):
            raise ValueError("Experiment 'splitMNIST' or 'cifar10' cannot have more than 10 tasks!")
        # configurations
        if name == 'splitMNIST':
            name = 'mnist'
        config = DATASET_CONFIGS[name]
        num_classes = config['classes']
        classes_per_task = int(np.floor(num_classes / tasks))
        # print("classes_per_task = " + str(classes_per_task))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutations = np.array(list(range(num_classes))) if exception else np.random.permutation(list(range(num_classes)))
            target_transform = transforms.Lambda(lambda y, x=permutations: int(permutations[y]))
            # prepare train and test datasets with all classes
            data_train = get_dataset(name, type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
            data_test = get_dataset(name, type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            # generate labels-per-task
            if super_class:
                labels_per_task = [
                    list(CIFAR_LABELS[task_id]) for task_id in range(tasks)
                ]
                target_dict = create_target_transform_dict(CIFAR_LABELS)
            else:
                labels_per_task = [
                    list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
                ]
            
            labels_per_task = labels_per_task[:new_nb_tasks]

            train_datasets = []
            test_datasets = []
            for idx, labels in enumerate(labels_per_task):
                print("idx:", idx, "labels:",labels)
                if super_class:
                    target_transform = transforms.Lambda(
                        lambda y, x=target_dict: x[y]
                    ) if scenario=='domain' else None
                else:
                    target_transform = transforms.Lambda(
                        lambda y, x=labels[0]: y - x
                    ) if scenario=='domain' else None
                # print("labels = " + str(labels))
                # print("len = " + str(len(SubDataset(data_train, labels, idx, classes_per_task, target_transform=target_transform))))
                train_datasets.append(SubDataset(data_train, labels, idx, classes_per_task, target_transform=target_transform))
                test_datasets.append(SubDataset(data_test, labels, idx, classes_per_task, target_transform=target_transform))
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task, permutations)


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self, sizes):
        super(MLP, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    def __init__(self, num_channels, output_size):
        super(Net, self).__init__()
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(self.num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
#         x = self.fc3(x)
        return x


def train(mynet, data, target, optimizer):
    data, target = data.to(device), target.to(device)
    mynet.train()
    output = mynet(data)
    loss = criterion(output, target)  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return


def test(mynet, data, target):
    mynet.eval()
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        output = mynet(data)
        loss = criterion(output, target)
        return loss

    
def test_task(net, dataloader):
    net.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for jdx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        print('Total accuracy: %0.2f %%' % (100.0 * correct / total))
    
class Expert:
    def __init__(self, net, net_id, num_classes_per_task, window_size, smoothing_factor):
        self.is_active = True
        self.net = net
        self.expert_id = net_id
        self.loss_window = deque(maxlen=window_size)
        self.loss_window.append(np.log(num_classes_per_task)) # initial value = entopy of random assignment
        
        # Smoothed loss when the expert is active
        self.smoothed_losses_when_active = []
        # self.smoothed_losses_when_active.append(np.log(num_classes_per_task)) 
        
        # complete track of loss, smoothed_loss, sd, avg, thresholds
        self.all_losses = []
        self.all_smoothed_losses = []
        self.all_STDs = []
        self.all_avges = []
        self.all_thresholds = []
        self.all_tasks = []

        self.smoothing_factor = smoothing_factor
        self.current_threshold = None
    
    def update_all_and_smoothed_losses(self, loss, task_id):
        self.all_losses.append(loss)
        last_smoothed_loss = loss if len(self.all_smoothed_losses) == 0 else self.all_smoothed_losses[-1]
        self.all_smoothed_losses.append(self.smoothing_factor * loss + (1 - self.smoothing_factor) * last_smoothed_loss)
        self.all_tasks.append(task_id)

    def get_current_smoothed_loss(self):
        # return self.smoothed_losses_when_active[-1]
        # print("debug smoothed_losses", self.all_smoothed_losses)
        return self.all_smoothed_losses[-1]

    def get_inactive_smoothed_loss(self, loss):
        last_smoothed_loss = loss if len(self.smoothed_losses_when_active) == 0 else self.smoothed_losses_when_active[-1]
        smoothed_loss = self.smoothing_factor * loss + (1 - self.smoothing_factor) * last_smoothed_loss
        return smoothed_loss

    def update_loss_window_when_active(self, loss):
        if self.is_active:
            self.loss_window.append(loss)
            last_smoothed_loss = loss if len(self.smoothed_losses_when_active) == 0 else self.smoothed_losses_when_active[-1]
            self.smoothed_losses_when_active.append(self.smoothing_factor * loss + \
                                                    (1 - self.smoothing_factor) * last_smoothed_loss)
            return True
        else:
            return False
       
    def get_threshold(self,loss):
        if self.is_active:
            # print("debug loss window", self.loss_window)
            avg_ = sum(np.asarray(self.loss_window)) / len(np.asarray(self.loss_window))
            sd_ = np.std(np.asarray(self.loss_window))
            self.current_threshold = avg_ + 3 * max(sd_, 0.005)

            self.all_avges.append(avg_)
            self.all_STDs.append(sd_)
            self.all_thresholds.append(self.current_threshold)            
            return self.current_threshold
        else:
            return self.current_threshold

def add_new_expert(dataset_name, num_channels, num_classes_per_task, learning_rate, optimizers, use_sch,
                   schedulers, warmup_schedulers, warmup_epochs, cur_iters, experts, current_expert_id, 
                   window_size, smoothing_factor, num_iter_per_epoch):
    if (dataset_name == 'cifar10' or dataset_name == 'cifar100'):
        # net = ResNet18(num_classes_per_task)
        net = VGG('VGG11', num_classes_per_task)
        # net = models.vgg11_bn(pretrained=True)
        # num_ftrs = net.classifier[6].in_features
        # net.classifier[6] = nn.Linear(num_ftrs, num_classes_per_task)

    else:
        net = Net(num_channels=num_channels, output_size=num_classes_per_task)      
        # net = MLP([784] + [100] * 2 + [num_classes_per_task])
        # net = MLP([n_inputs] + [nh] * nl + [n_outputs])
    net = net.to(device)
    
    # debug, to compute model size
    # param = sum([p.numel() for p in net.parameters()])
    # print("debug model size", param)      
    
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizers.append(optimizer)
    if use_sch == 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)
    elif use_sch == 2:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)
    schedulers.append(scheduler)
    if warmup_epochs > 0:
        warmup_scheduler = WarmUpLR(optimizer, warmup_epochs * num_iter_per_epoch)
        warmup_schedulers.append(warmup_scheduler)
    cur_iters.append(0)
    experts.append(Expert(net=net, net_id=current_expert_id, num_classes_per_task=num_classes_per_task, 
                          window_size=window_size, smoothing_factor=smoothing_factor))


def train_wrapper(experts, current_expert_id, data, target, optimizers, cur_iters, use_sch, schedulers, 
                  warmup_schedulers, warmup_epochs, losses, num_iter_per_epoch):
    train(experts[current_expert_id].net, data, target, optimizers[current_expert_id])
    cur_iters[current_expert_id] += 1
    if (warmup_epochs > 0 and cur_iters[current_expert_id] < warmup_epochs * num_iter_per_epoch):
        warmup_schedulers[current_expert_id].step()
        # print("iter warm = " + str(cur_iters[current_expert_id]))
        # print("lr = ", optimizers[current_expert_id].param_groups[0]['lr'])
    if (use_sch and cur_iters[current_expert_id] % num_iter_per_epoch == 0):
        schedulers[current_expert_id].step()
        # print("iter = " + str(cur_iters[current_expert_id]))
        # print("lr = ", optimizers[current_expert_id].param_groups[0]['lr'])
    experts[current_expert_id].update_loss_window_when_active(losses[current_expert_id])

    
    
def gradual_prunning(experts, current_expert_id, prune_dataloader, use_sch, conv2d_prune_amount=0.98, linear_prune_amount=0.98):
    expert = experts[current_expert_id].net
    print("performance before prunning")
    test_task(expert, prune_dataloader)

    l1_regularization_strength = 0
    l2_regularization_strength = 1e-4
    learning_rate = 1e-1
    learning_rate_decay = 1

    
    model_dir = "saved_models"
    model_filename_prefix = "pruned_model_expert_"+str(current_expert_id)
    pruned_model_filename = "expert_"+str(current_expert_id)+"_pruned_cifar100.pt"
    pruned_model_filepath = os.path.join(model_dir, pruned_model_filename)

    iterative_pruning_finetuning(
            model=expert,
            train_loader=prune_dataloader,
            test_loader=prune_dataloader,
            device=device,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            l1_regularization_strength=l1_regularization_strength,
            l2_regularization_strength=l2_regularization_strength,
            conv2d_prune_amount=conv2d_prune_amount,
            linear_prune_amount=linear_prune_amount,
            num_iterations=1,
            num_epochs_per_iteration=200,
            model_filename_prefix=model_filename_prefix,
            model_dir=model_dir,
            grouped_pruning=True)
    
    print("performance after prunning")
    test_task(expert, prune_dataloader)
    print("save current model")
    experts[current_expert_id].net = expert
    print("sparsity")
    
def train_all_experts(num_channels, num_classes_per_task, replay_capacity, prune_capacity, learning_rate, batch_size, dataset, 
                      dataloader, window_size, smoothing_factor, use_sch, num_iter_per_epoch, warmup_epochs):

    dataset_name = 'mnist' if (dataset == 'splitMNIST' or dataset == 'permMNIST') else dataset
    num_classes = DATASET_CONFIGS[dataset_name]['classes']
    if dataset == 'permMNIST':
        num_classes *= 20
    num_experts = 0
    current_expert_id = None
    scheduler = None
    prev_task_id = 0
    experts = []
    optimizers = []
    schedulers = []
    warmup_schedulers = []
    cur_iters = []
    indices_set = set()
    list_pqs = []
    for _ in range(num_classes):
        pq = []
        heapq.heapify(pq)
        list_pqs.append(pq)
        
    # new feature: priorty queue for prunning
    prune_pq = []
    heapq.heapify(prune_pq)
    
    # for experiments without smoothing
    task_iteration_list = []
    expert_iteration_list = []
    
    print("total iterations", len(dataloader))
    for jdx, (data, target, task_id, class_id, index) in enumerate(dataloader):
        if jdx % 100 ==0:
            print("current iteration", jdx)
        data, target = data.to(device), target.to(device)
        if task_id[0].item() != prev_task_id:
            print('Task has changed to {}, at iter {}'.format(task_id[0].item(), jdx))
            prev_task_id = task_id[0].item()
        if len(experts) == 0:
            print('Added new expert at iter '+ str(jdx))
            current_expert_id = num_experts
            print('current expert id', current_expert_id)
            add_new_expert(dataset_name, num_channels, num_classes_per_task, learning_rate, optimizers, use_sch,
                           schedulers, warmup_schedulers, warmup_epochs, cur_iters, experts, current_expert_id, 
                           window_size, smoothing_factor, num_iter_per_epoch)
            num_experts += 1 

        losses = []
        for expert in experts:
            loss = test(expert.net, data, target).detach().cpu().numpy()
            losses.append(loss)
            expert.update_all_and_smoothed_losses(loss, task_id[0].item())
        
        thresholds = []
        for idx, expert in enumerate(experts):
            threshold = expert.get_threshold(losses[idx])
            thresholds.append(threshold)
        
        if experts[current_expert_id].get_current_smoothed_loss() > thresholds[current_expert_id]:
            experts[current_expert_id].is_active = False
            current_expert_id = None
            print("Time to switch at iter " + str(jdx))
            for idx, expert in enumerate(experts):
                if expert.get_inactive_smoothed_loss(losses[idx]) < thresholds[idx]:
                    print("Switching to the expert " + str(idx))
                    current_expert_id = idx
                    experts[current_expert_id].is_active = True
                    train_wrapper(experts, current_expert_id, data, target, optimizers, cur_iters, use_sch, schedulers, 
                                  warmup_schedulers, warmup_epochs, losses, num_iter_per_epoch)
                    break
                    
            if current_expert_id == None:
                print("Added a new expert at iter " + str(jdx))
                current_expert_id = num_experts
                print("Number of experts:", num_experts)
                add_new_expert(dataset_name, num_channels, num_classes_per_task, learning_rate, optimizers, use_sch,
                               schedulers, warmup_schedulers, warmup_epochs, cur_iters, experts, current_expert_id, 
                               window_size, smoothing_factor, num_iter_per_epoch)
                num_experts += 1
                losses.append(test(experts[current_expert_id].net, data, target).detach().cpu().numpy())
                train_wrapper(experts, current_expert_id, data, target, optimizers, cur_iters, use_sch, schedulers, 
                              warmup_schedulers, warmup_epochs, losses, num_iter_per_epoch)
                
                # new feature: gradual prunning for current expert
                list_prune_data_indexes = []
                list_prune_new_targets = []
                for _ in range(len(prune_pq)):
                    _, (index_prune, targ_prune) = heapq.heappop(prune_pq)
                    list_prune_data_indexes.append(index_prune)
                    list_prune_new_targets.append(targ_prune)
                
                if (dataset == 'cifar10' or dataset == 'cifar100'):
                    data_train = get_dataset(dataset_name, type="train", dir='./datasets', target_transform=None, verbose=False)
                else:
                    new_data = torch.Tensor(list_prune_data_indexes)
                    new_target = torch.Tensor(list_prune_new_targets).type(torch.LongTensor)
                prune_pq = []
                heapq.heapify(prune_pq)
                
                
        else:
            train_wrapper(experts, current_expert_id, data, target, optimizers, cur_iters, use_sch, schedulers, 
                          warmup_schedulers, warmup_epochs, losses, num_iter_per_epoch)

            
        task_iteration_list.append(task_id)
        expert_iteration_list.append(current_expert_id)
        if jdx % 100 ==0:
            print("length", len(expert_iteration_list))
            print("expert iteration list", expert_iteration_list[-100:-1])
        
        
        
        for idx, dat in enumerate(data.cpu().numpy()):
            cur_class_id = class_id[idx].item()
            cur_index = index[idx].item()
            cur_target = target[idx].item()
            if (cur_class_id, cur_index) not in indices_set:
                indices_set.add((cur_class_id, cur_index))
                priority = random.uniform(0, 1)
                if (dataset == 'cifar10' or dataset == 'cifar100'):
                    pair = (cur_index, current_expert_id)
                    prune_pair = (cur_index, cur_target)
                elif (dataset == 'splitMNIST' or dataset == 'permMNIST'):
                    pair = (dat, current_expert_id)
                    prune_pair = (dat, cur_target)
                else:
                    print("unknown dataset")
                heapq.heappush(list_pqs[cur_class_id], (priority, pair))
                # new feature
                heapq.heappush(prune_pq, (priority, prune_pair))
                if len(list_pqs[cur_class_id]) > int(replay_capacity / num_classes):
                    heapq.heappop(list_pqs[cur_class_id])[1][1] 
                if len(prune_pq) > prune_capacity:
                    heapq.heappop(prune_pq)[1][1]     
    
    list_data_indexes = []
    list_new_targets= []   
    for i in range(num_classes):
        for _ in range(len(list_pqs[i])):
            _, (index, targ) = heapq.heappop(list_pqs[i])
            list_data_indexes.append(index)
            list_new_targets.append(targ)

    if (dataset == 'cifar10' or dataset == 'cifar100'):
        data_train = get_dataset_buffer(dataset_name, type=type, dir='./datasets', target_transform=None, verbose=False)
        new_dataloader = DataLoader(CustomBufferDataset(data_train, list_data_indexes, list_new_targets), batch_size=batch_size, shuffle=True)
    else:  # MNIST TODO
        new_data = torch.Tensor(list_data_indexes)
        new_target = torch.Tensor(list_new_targets).type(torch.LongTensor)
        new_dataloader = DataLoader(TensorDataset(new_data, new_target), batch_size=batch_size, shuffle=True)
  
    return experts, new_dataloader


def train_expert_selector(net, learning_rate, dataloader, epochs, use_sch, warmup_epochs):
    losses = []
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    if use_sch == 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)
    elif use_sch == 2:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=1) #TODO GAMMA = 1
    if warmup_epochs > 0:
        num_iter_per_epoch = len(dataloader)
        warmup_scheduler = WarmUpLR(optimizer, warmup_epochs * num_iter_per_epoch)
    for epoch in range(epochs):
        for jdx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            train(net, data, target, optimizer)
            losses.append(test(net, data, target).detach().cpu().numpy())
            if (warmup_epochs > 0 and epoch < warmup_epochs):
                warmup_scheduler.step()
        if use_sch:
            scheduler.step()
            if epoch % 10 == 0:
                print("selector epoch = " + str(epoch))
                print("lr = ", optimizer.param_groups[0]['lr'])

    
    # prune expert selector
    l1_regularization_strength = 0
    l2_regularization_strength = 1e-4
    learning_rate = 1e-3
    learning_rate_decay = 1

    model_dir = "saved_models"
    model_filename_prefix = "pruned_model_expert_selector"
    pruned_model_filename = "expert_selector.pt"
    pruned_model_filepath = os.path.join(model_dir, pruned_model_filename)

    iterative_pruning_finetuning(
            model=net,
            train_loader=dataloader,
            test_loader=dataloader,
            device=device,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            l1_regularization_strength=l1_regularization_strength,
            l2_regularization_strength=l2_regularization_strength,
            conv2d_prune_amount=0.5,
            linear_prune_amount=0.5,
            num_iterations=1,
            num_epochs_per_iteration=200,
            model_filename_prefix=model_filename_prefix,
            model_dir=model_dir,
            grouped_pruning=True)
    
    print("performance after prunning")
    test_task(net, dataloader)
    
    return losses


def acc_on_expert_selector_train(net, num_experts, dataloader):
    net.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for kdx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print('\tTotal accuracy of expert selector: %0.2f %%' % (100.0 * correct / total))  


def acc_on_expert_selector_test(name, net, num_experts, dataloader):
    net.eval()
    correct = 0.0
    total = 0.0
    if (name == 'cifar10' or name == 'cifar100'):
        with torch.no_grad():
            for kdx, (data, _, _, targets, _) in enumerate(dataloader):
                data, targets = data.to(device), targets.to(device)
                outputs = net(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item() 
    else:
        with torch.no_grad():
            for kdx, (data, targets) in enumerate(dataloader):
                data, targets = data.to(device), targets.to(device)
                outputs = net(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item() 
    print('\tTotal accuracy of expert selector: %0.2f %%' % (100.0 * correct / total))  

                
def acc_on_class(expert, idx, num_tasks, num_classes_per_task, dataset, jdx):
    testloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=num_tasks)
    expert_net = expert.net
    expert_net.eval()
    correct = 0.0
    total = 0.0
    class_correct = list(0. for i in range(num_classes_per_task))
    class_total = list(0. for i in range(num_classes_per_task))
    with torch.no_grad(): 
        for kdx, (data, targets, task_id, class_id, index) in enumerate(testloader):
            data, targets = data.to(device), targets.to(device)
            outputs = expert_net(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            c = (predicted == targets)
            for idxx, target in enumerate(targets):
                class_correct[target] += c[idxx].item()
                class_total[target] += 1
    print('\nTesting stats for task ' + str(jdx+1) + ' and expert ' + str(idx+1))
    for i in range(num_classes_per_task):
        print('\tAccuracy on %5s : %0.2f %%' % ('Class ' + str(i + 1), 100.0 * class_correct[i] / class_total[i]))
    print('\tTotal accuracy: %0.2f %%' % (100.0 * correct / total))  


def acc_total(experts, expert_selector, num_classes_per_task, testloader):
    for i in range(len(experts)):
        experts[i].net.eval()
    expert_selector.eval()
    correct = 0.0
    total = 0.0
    correct_ic = 0.0
    with torch.no_grad():
        for jdx, (data, target, task_id, class_id, index) in enumerate(testloader):
            data, target, task_id, class_id = data.to(device), target.to(device), task_id.to(device), class_id.to(device)
            outputs = expert_selector(data)
            _, predicted_expert_id = torch.max(outputs.data, 1)
            selected_expert = experts[predicted_expert_id].net
            outputs = selected_expert(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item() 
            predicted += num_classes_per_task * predicted_expert_id
            correct_ic += (predicted == class_id).sum().item()

    print('Total accuracy: %0.2f %%' % (100.0 * correct / total))   
    # print('Total accuracy IC: %0.2f %%' % (100.0 * correct_ic / total))  


def acc_total_new(experts, expert_selector, num_classes_per_task, testloader):
    for i in range(len(experts)):
        experts[i].net.eval()
    expert_selector.eval()
    correct = 0.0
    total = 0.0
    correct_sel = 0.0
    total_sel = 0.0
    correct_ic = 0
    with torch.no_grad():
        for jdx, (selector_data, data, targets, task_id, class_id) in enumerate(testloader):
            selector_data, data, targets, task_id, class_id = selector_data.to(device), data.to(device), targets.to(device), task_id.to(device), class_id.to(device)
            outputs = expert_selector(selector_data)
            _, predicted_expert_id = torch.max(outputs.data, 1)
            correct_sel += (predicted_expert_id == task_id).sum().item()
            total_sel += task_id.size(0)
            selected_expert = experts[predicted_expert_id].net  # TODO
            outputs = selected_expert(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            predicted += num_classes_per_task * predicted_expert_id
            correct_ic += (predicted == class_id).sum().item()

    print('Total accuracy: %0.2f %%' % (100.0 * correct / total))     
    # print('Total accuracy IC: %0.2f %%' % (100.0 * correct_ic / total))   
    # print('Selector accuracy again: %0.2f %%' % (100.0 * correct_sel / total_sel))     
      

def create_selector_dataloader(name, type, replay_capacity, num_classes, batch_size, dataloader):
    list_pqs = []
    for _ in range(num_classes):
        pq = []
        heapq.heapify(pq)
        list_pqs.append(pq)   
        
    for jdx, (data, target, task_id, class_id, index) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        for idx, dat in enumerate(data.cpu().numpy()):
            priority = random.uniform(0, 1)
            if (name == 'cifar10' or name == 'cifar100'):
                pair = (index[idx].item(), task_id[idx].item())
            else:
                pair = (dat, task_id[idx].item())
            heapq.heappush(list_pqs[class_id[idx].item()], (priority, pair))
            if len(list_pqs[class_id[idx].item()]) > int(replay_capacity / num_classes):
                heapq.heappop(list_pqs[class_id[idx].item()])[1][1]
    
    list_data_indexes = []
    list_new_targets= []   
    for i in range(num_classes):
        for _ in range(len(list_pqs[i])):
            _, (index, targ) = heapq.heappop(list_pqs[i])
            list_data_indexes.append(index)
            list_new_targets.append(targ)

    # if (name == 'splitMNIST' or name == 'permMNIST'):
    #     name = 'mnist'

    if (name == 'cifar10' or name == 'cifar100'):
        data_train = get_dataset_buffer(name, type=type, dir='./datasets', target_transform=None, verbose=False)
        new_dataloader = DataLoader(CustomBufferDataset(data_train, list_data_indexes, list_new_targets), batch_size=batch_size, shuffle=True)
    else:
        new_data = torch.Tensor(list_data_indexes)
        new_target = torch.Tensor(list_new_targets).type(torch.LongTensor)
        new_dataloader = DataLoader(TensorDataset(new_data, new_target), batch_size=batch_size, shuffle=True)

    return new_dataloader


def create_test_dataloader(name, type, replay_capacity, num_classes, batch_size, dataloader):
    list_data = []
    list_data_indexes = []
    list_targets = []
    list_task_ids = []
    list_class_ids = []
    for jdx, (data, target, task_id, class_id, index) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        for idx, dat in enumerate(data.cpu().numpy()):
            if (name == 'cifar10' or name == 'cifar100'):
                list_data_indexes.append(index[idx].item())
            else:
                list_data.append(dat)
            list_targets.append(target[idx].item())
            list_task_ids.append(task_id[idx].item())
            list_class_ids.append(class_id[idx].item())

    # if (name == 'splitMNIST' or name == 'permMNIST'):
    #     data_name = 'mnist'

    if (name == 'cifar10' or name == 'cifar100'):
        data_for_selector = get_dataset_buffer(name, type="test", dir='./datasets', target_transform=None, verbose=False)
        data_for_experts = get_dataset(name, type="test", dir='./datasets', target_transform=None, verbose=False)
        new_dataloader = DataLoader(CustomTestDataset(data_for_selector, data_for_experts, list_data_indexes, list_targets, 
            list_task_ids, list_class_ids), batch_size=batch_size, shuffle=True)
    else:
        new_data = torch.Tensor(list_data)
        new_target = torch.Tensor(list_task_ids).type(torch.LongTensor)
        new_dataloader = DataLoader(TensorDataset(new_data, new_target), batch_size=batch_size, shuffle=True)

    return new_dataloader


def write_file(name, experts, smoothed):
    name = 's_losses_' + name if smoothed else 'losses_' + name
    with open('results/' + name + '.txt', 'w') as output:
        for i in range(len(experts)):
            losses = experts[i].all_losses if not smoothed else experts[i].all_smoothed_losses
            for j in range(len(losses)):
                output.write(str(losses[j]) + ' ')
            output.write('\n')
    
    name = 'tasks_id_' + name 
    with open('results/' + name + '.txt', 'w') as output:
        tasks = experts[0].all_tasks
        for j in range(len(tasks)):
            output.write(str(tasks[j]) + ' ')


def read_file(name, smoothed):
    name = 's_losses_' + name if smoothed else 'losses_' + name
    losses = []
    with open('results/' + name + '.txt') as f:
        for line in f:
            line = line.split() # to deal with blank 
            if line:            # lines (ie skip them)
                line = [float(i) for i in line]
                losses.append(line)

    tasks = []
    name = 'tasks_id_' + name 
    with open('results/' + name + '.txt') as f:
        for line in f:
            line = line.split() # to deal with blank 
            if line:            # lines (ie skip them)
                line = [int(i) for i in line]
                tasks.extend(line)


    return losses, tasks
    

def plot_figures(name, losses, tasks, smoothed):
    fig, ax = plt.subplots()
    for i in range(len(losses)):
        ax.plot(losses[i])
    
    legend = ['Expert ' + str(i + 1) for i in range(len(losses))]
    lgd = ax.legend(legend, loc='upper center', bbox_to_anchor= (1.15, 0.9))
    for line in lgd.get_lines():
        line.set_linewidth(4)
    
    # tasks = [1 + t for t in tasks]
    # ax2 = ax.twinx()
    # ax2.plot(tasks, color = 'black', label = 'Task ID', linewidth=3)
    # lgd2 = ax2.legend(loc='upper center', bbox_to_anchor= (1.15, 1))
    # ax2.set_ylabel("Task ID")
    # ax2.set_yticks([i + 1 for i in range(len(losses))])
    # for line in lgd2.get_lines():
    #     line.set_linewidth(4)
    
    if smoothed:
        ax.set_ylabel('Smoothed Loss')
    else:
        ax.set_ylabel('Loss')
    plt.xlabel('Iterations')

    name = 's_losses_' + name if smoothed else 'losses_' + name
    fig.savefig('results/' + name + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
