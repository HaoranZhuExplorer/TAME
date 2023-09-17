# test trained models

import argparse
import copy
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import ConcatDataset, Dataset, TensorDataset, DataLoader
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import math
from collections import deque
import heapq
import random
from models import *
from utils_ import *
from prune_networks import prune_network


torch.multiprocessing.set_sharing_strategy('file_system')  # to solve memory leak issue

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='split-mnist',
                   help='Data set name')
parser.add_argument('--num_tasks', type=int, default=5,
                   help='Num tasks for dataset')

parser.add_argument('--cap', type=int, default=5000,
                   help='Buffer capacity')

def main():
    PRE_TRAINED = True
    ONLY_PLOT = False # This line or below should be True for Fig. 1, 2, 4.
    NEW_PLOT = True 

    args = parser.parse_args()

    if args.data == 'cifar100':
        DATASET_NAME = 'cifar100'
        WARMUP_EPOCHS = 1
        USE_SCH = 2
        ONLY_SELECTOR = False
        SUPER_CLASS = False
        NUM_TASKS = 20
        NUM_TASKS_NEW = args.num_tasks
        NUM_CLASSES_PER_TASK = int(100 / NUM_TASKS)
        BATCH_SIZE = 128
        SMOOTHING_FACTOR = 0.2
        WINDOW_SIZE = 100
        REPLAY_CAPACITY = args.cap
        EPOCHS = 200
        LEARNING_RATE = 0.1 
        SELECTOR_EPOCHS = 200
        SELECTOR_LEARNING_RATE = 0.0001
        SELECTOR_BATCH_SIZE = 128

    elif args.data == 'split-mnist':
        DATASET_NAME = 'splitMNIST'
        WARMUP_EPOCHS = 0
        USE_SCH = 2
        ONLY_SELECTOR = False
        SUPER_CLASS = False
        NUM_TASKS = 5
        NUM_TASKS_NEW = args.num_tasks
        NUM_CLASSES_PER_TASK = int(10 / NUM_TASKS)
        BATCH_SIZE = 128
        SMOOTHING_FACTOR = 0.2
        WINDOW_SIZE = 35
        REPLAY_CAPACITY = args.cap
        EPOCHS = 10
        LEARNING_RATE = 0.1
        SELECTOR_EPOCHS = 100
        SELECTOR_LEARNING_RATE = 0.1
        SELECTOR_BATCH_SIZE = 128

    elif args.data == 'perm-mnist':
        DATASET_NAME = 'permMNIST'
        WARMUP_EPOCHS = 0
        USE_SCH = 2
        ONLY_SELECTOR = False
        SUPER_CLASS = False
        NUM_TASKS = args.num_tasks
        NUM_TASKS_NEW = args.num_tasks
        NUM_CLASSES_PER_TASK = 10
        BATCH_SIZE = 128 
        SMOOTHING_FACTOR = 0.2
        WINDOW_SIZE = 100
        REPLAY_CAPACITY = args.cap
        EPOCHS = 10
        LEARNING_RATE = 0.1
        SELECTOR_EPOCHS = 100
        SELECTOR_LEARNING_RATE = 0.1
        SELECTOR_BATCH_SIZE = 128

    elif args.data == 'cifar10':
        DATASET_NAME = 'cifar10'
        WARMUP_EPOCHS = 0
        USE_SCH = 2
        ONLY_SELECTOR = False
        SUPER_CLASS = False
        NUM_TASKS = 5
        NUM_TASKS_NEW = args.num_tasks
        NUM_CLASSES_PER_TASK = int(10 / NUM_TASKS)
        BATCH_SIZE = 128
        SMOOTHING_FACTOR = 0.1
        WINDOW_SIZE = 100
        REPLAY_CAPACITY = 5000
        EPOCHS = 200
        LEARNING_RATE = 0.1
        SELECTOR_EPOCHS = 200
        SELECTOR_LEARNING_RATE = 0.0001
        SELECTOR_BATCH_SIZE = 100
    
    else:
        print("Not valid dataset")
        exit()
   
    if ONLY_PLOT:
        losses, tasks = read_file(DATASET_NAME, False)
        smoothed_losses, _ = read_file(DATASET_NAME, True)
        plot_figures(DATASET_NAME, losses, tasks, False)
        plot_figures(DATASET_NAME, smoothed_losses, tasks, True)
        exit()

    (train_datasets, test_datasets), config, classes_per_task, permutations = get_multitask_experiment(
        name=DATASET_NAME, scenario='domain', tasks=NUM_TASKS, new_nb_tasks = NUM_TASKS_NEW, data_dir='./datasets',
        verbose=False, exception=True if 0 == 0 else False, super_class=SUPER_CLASS
    )

    for i in range(len(train_datasets)):
        print("len(train_datasets[i]) = ", len(train_datasets[i]))

    if ONLY_SELECTOR:
        custom_list = [i for i in range(NUM_TASKS_NEW)]
    else:
        custom_list = [i for i in range(NUM_TASKS_NEW) for _ in range(EPOCHS)]

    custom_list = [0 for _ in range(EPOCHS)] + [1 for _ in range(EPOCHS)] + \
        [2 for _ in range(EPOCHS)] + [1 for _ in range(EPOCHS)] + [3 for _ in range(EPOCHS)] # Uncomment for Fig. 4b

    custom_list += custom_list # Uncomment for Fig. 4a
    custom_data_train = []
    for value in custom_list:
        custom_data_train.append(train_datasets[value])
    custom_dataset_train = torch.utils.data.ConcatDataset(custom_data_train)

    trainloader = torch.utils.data.DataLoader(custom_dataset_train, batch_size=BATCH_SIZE, shuffle=False)
    dataset_name = 'mnist' if (DATASET_NAME == 'splitMNIST' or DATASET_NAME == 'permMNIST') else DATASET_NAME

    num_classes = NUM_TASKS_NEW * NUM_CLASSES_PER_TASK
    if ONLY_SELECTOR:
        expert_selector_trainloader = create_selector_dataloader(name=DATASET_NAME, type="train", replay_capacity=REPLAY_CAPACITY, num_classes=num_classes, batch_size=SELECTOR_BATCH_SIZE, dataloader=trainloader)
        num_experts = NUM_TASKS_NEW
    else:
        num_iter_per_epoch = int(np.ceil(len(train_datasets[0]) / BATCH_SIZE))
        print("num_iter_per_epoch = ", num_iter_per_epoch)
        
#         experts, expert_selector_trainloader = train_all_experts(num_channels=DATASET_CONFIGS[dataset_name]['channels'], 
#                                                                 num_classes_per_task=NUM_CLASSES_PER_TASK, 
#                                                                 replay_capacity=REPLAY_CAPACITY, learning_rate=LEARNING_RATE,
#                                                                 batch_size=SELECTOR_BATCH_SIZE, dataset=DATASET_NAME,
#                                                                 dataloader=trainloader, window_size=WINDOW_SIZE, 
#                                                                 smoothing_factor=SMOOTHING_FACTOR, use_sch=USE_SCH, 
#                                                                 num_iter_per_epoch=num_iter_per_epoch,
#                                                                 warmup_epochs=WARMUP_EPOCHS)
#        num_experts = len(experts)
    
#     torch.cuda.empty_cache()

#     if (DATASET_NAME == 'cifar10' or DATASET_NAME == 'cifar100'):
#         if PRE_TRAINED:
#             expert_selector = models.resnet18(pretrained=True) #densenet161
#             # for param in expert_selector.parameters():
#             #     param.requires_grad = False
#             num_ftrs = expert_selector.fc.in_features
#             expert_selector.fc = nn.Linear(num_ftrs, num_experts)
#         else:    
#             expert_selector = ResNet18_2(num_classes=num_experts)
#     else:
#         expert_selector = Net(num_channels=DATASET_CONFIGS[dataset_name]['channels'], output_size=num_experts)
        
#     expert_selector = expert_selector.to(device)
#     if device == 'cuda':
#         expert_selector = torch.nn.DataParallel(expert_selector)
#         cudnn.benchmark = True

    print("Dataset = ", DATASET_NAME)
    print("Number of tasks = ", NUM_TASKS_NEW)
    print("Buffer capacity = ", REPLAY_CAPACITY)

    



    testdatas = []
    for i in range(NUM_TASKS_NEW):
        testdatas.append(test_datasets[i])
    testdataset = torch.utils.data.ConcatDataset(testdatas)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=NUM_TASKS_NEW)
    testloader_selector = create_test_dataloader(name=DATASET_NAME, type="test", replay_capacity=1e7, num_classes=num_classes, batch_size=1, dataloader=testloader)  # TODO
    
    expert_selector = torch.load("saved/expert_selector_cap20000.pt")
    print("Accuracy of expert selector on test data:")
    acc_on_expert_selector_test(name=DATASET_NAME, net=expert_selector, num_experts=NUM_TASKS_NEW, dataloader=testloader_selector)
            

    if (DATASET_NAME == 'cifar10' or DATASET_NAME == 'cifar100'):
        for ratio in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            print("prunning ratio", ratio)
            experts = torch.load("saved/experts_cap20000.pt")
            for i in range(len(experts)):
                prune_network(experts[i], prunning_ratio=ratio)

            # print("Accuracy of experts on each task of test data")
            for jdx, task in enumerate(test_datasets):
                for idx, expert in enumerate(experts):
                    if jdx == idx:
                        acc_on_class(expert, idx, NUM_TASKS_NEW, NUM_CLASSES_PER_TASK, task, jdx)

            print("computing average accuracy")
            acc_total_new(experts, expert_selector, NUM_CLASSES_PER_TASK, testloader_selector)
    else:
        acc_total(experts, expert_selector, NUM_CLASSES_PER_TASK, testloader)
        

        

        
if __name__ == '__main__':
    main()
