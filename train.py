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

torch.multiprocessing.set_sharing_strategy('file_system')  # to solve memory leak issue

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='split-mnist',
                   help='Data set name')
parser.add_argument('--num_tasks', type=int, default=5,
                   help='Num tasks for dataset')
parser.add_argument('--cap', type=int, default=5000,
                   help='Buffer capacity')
parser.add_argument('--prune_cap', type=int, default=500,
                   help='Prunning Buffer capacity')



def main():
    PRE_TRAINED = True
    ONLY_PLOT = False # This line or below should be True for Fig. 1, 2, 4.
    NEW_PLOT = True 

    args = parser.parse_args()

    if args.data == 'split-mnist':
        DATASET_NAME = 'splitMNIST'
        WARMUP_EPOCHS = 0
        USE_SCH = 2
        ONLY_SELECTOR = False
        SUPER_CLASS = False
        NUM_TASKS = 5
        NUM_TASKS_NEW = args.num_tasks
        NUM_CLASSES_PER_TASK = int(10 / NUM_TASKS)
        BATCH_SIZE = 128
        SMOOTHING_FACTOR = 0.2 # 0.2, original setting
        WINDOW_SIZE = 35
        REPLAY_CAPACITY = args.cap
        PRUNE_CAPACITY = args.prune_cap
        EPOCHS = 10
        LEARNING_RATE = 0.1
        SELECTOR_EPOCHS = 200
        SELECTOR_LEARNING_RATE = 0.1
        SELECTOR_BATCH_SIZE = 128
    
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
        
        
    if ONLY_SELECTOR:
        custom_list = [i for i in range(NUM_TASKS_NEW)]
    else:
        custom_list = [i for i in range(NUM_TASKS_NEW) for _ in range(EPOCHS)]

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
        experts, expert_selector_trainloader = train_all_experts(num_channels=DATASET_CONFIGS[dataset_name]['channels'], 
                                                                num_classes_per_task=NUM_CLASSES_PER_TASK, 
                                                                replay_capacity=REPLAY_CAPACITY, prune_capacity = PRUNE_CAPACITY, learning_rate=LEARNING_RATE, 
                                                                batch_size=SELECTOR_BATCH_SIZE, dataset=DATASET_NAME,
                                                                dataloader=trainloader, window_size=WINDOW_SIZE, 
                                                                smoothing_factor=SMOOTHING_FACTOR, use_sch=USE_SCH, 
                                                                num_iter_per_epoch=num_iter_per_epoch,
                                                                warmup_epochs=WARMUP_EPOCHS)
        num_experts = len(experts)
    
    torch.cuda.empty_cache()


    expert_selector = Net(num_channels=DATASET_CONFIGS[dataset_name]['channels'], output_size=num_experts)    
    expert_selector = expert_selector.to(device)
    if device == 'cuda':
        expert_selector = torch.nn.DataParallel(expert_selector)
        cudnn.benchmark = True

    param = sum([p.numel() for p in expert_selector.parameters()])
    print("selector network size", param)
    
    print("train selector network")
    selector_losses = train_expert_selector(expert_selector, SELECTOR_LEARNING_RATE, expert_selector_trainloader, epochs=SELECTOR_EPOCHS, use_sch=USE_SCH, warmup_epochs=WARMUP_EPOCHS) 
    plt.rcParams["figure.figsize"] = (15,10)
    fig, ax = plt.subplots()
    ax.plot(selector_losses)
    ax.legend(["Selector loss"])
    fig.savefig('results/selector_' + DATASET_NAME + '.png')

    print("Dataset = ", DATASET_NAME)
    print("Number of tasks = ", NUM_TASKS_NEW)
    print("Number of experts = ", num_experts)
    print("Buffer capacity = ", REPLAY_CAPACITY)

    testdatas = []
    for i in range(NUM_TASKS_NEW):
        testdatas.append(test_datasets[i])
    testdataset = torch.utils.data.ConcatDataset(testdatas)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=NUM_TASKS_NEW)

    testloader_selector = create_test_dataloader(name=DATASET_NAME, type="test", replay_capacity=1e7, num_classes=num_classes, batch_size=1, dataloader=testloader)  # TODO

    print("Accuracy of expert selector on test data:")
    acc_on_expert_selector_test(name=DATASET_NAME, net=expert_selector, num_experts=NUM_TASKS_NEW, dataloader=testloader_selector)

    if not ONLY_SELECTOR:  
        if NEW_PLOT:
            for expert in experts:
                expert.all_losses = np.pad(np.array(expert.all_losses), (len(experts[0].all_losses)-len(expert.all_losses), 0), mode='edge')
                expert.all_smoothed_losses = np.pad(np.array(expert.all_smoothed_losses), (len(experts[0].all_smoothed_losses)-len(expert.all_smoothed_losses), 0), mode='edge')
            
            write_file(DATASET_NAME, experts, False)
            write_file(DATASET_NAME, experts, True)

            losses, tasks = read_file(DATASET_NAME, False)
            smoothed_losses, _ = read_file(DATASET_NAME, True)

            plot_figures(DATASET_NAME, losses, tasks, False)
            plot_figures(DATASET_NAME, smoothed_losses, tasks, True)

        print("Accuracy of experts on each task of test data")
        for jdx, task in enumerate(test_datasets):
            for idx, expert in enumerate(experts):
                if jdx == idx:
                    acc_on_class(expert, idx, NUM_TASKS_NEW, NUM_CLASSES_PER_TASK, task, jdx)

        
        print("computing average accuracy")
        acc_total(experts, expert_selector, NUM_CLASSES_PER_TASK, testloader)
        torch.save(experts, "saved/"+DATASET_NAME+"/experts_"+str(PRUNE_CAPACITY)+".pt")
        torch.save(expert_selector, "saved/"+DATASET_NAME+"/expert_selector_"+str(PRUNE_CAPACITY)+".pt")

        
if __name__ == '__main__':
    main()
