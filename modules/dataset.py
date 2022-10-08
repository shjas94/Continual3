import os
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image



def _get_permuted_img(img, permutation):
    if permutation is None:
        return img
    
    c, h, w = img.size()
    img = img.view(-1, c)
    img = img[permutation, :]
    return img.view(c, h, w)

def get_transforms(args):
    if args.dataset == 'splitted_mnist':
        return transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
    elif args.dataset == 'permuted_mnist':
        permutation = [np.random.permutation(28**2) for _ in range(args.num_tasks)]
        return transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,)),
                                 transforms.Lambda(lambda x:_get_permuted_img(x, permutation))
                                 ])
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        return transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
                                 ])
    elif args.dataset == 'oxford_pet':
        return transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Resize((128, 128)), # To Do : img size argparse로 입력받기
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                 ])
    
class Oxford_Pet(data.Dataset):
    def __init__(self, root, data_path, transforms=None, train=True) -> None:
        super().__init__()
        self.root = root
        self.data_path = os.listdir(os.path.join(self.root,data_path))
        self.data_path = self._sort_by_id(self.data_path) # ex) images/beagle_0.jpg
        self.data_path = self.delete_mat_format()   
        self.label = self._make_label()
        self.transforms = transforms
        self.train = train
        
        self.train_path, \
        self.test_path, \
        self.train_label, \
        self.test_label = train_test_split(self.data_path,
                                           self.label, 
                                           test_size=0.25, 
                                           random_state=42,
                                           shuffle=True,
                                           stratify=self.label)
    def delete_mat_format(self):
        new_data_path = []
        for i in range(len(self.data_path)):
            if self.data_path[i].split('.')[-1] == 'mat':
                continue
            new_data_path.append(self.data_path[i])
        return new_data_path
    def _sort_by_id(self, data_path):
        new_data_path = sorted(data_path, key=lambda x: x.split('.')[0].split('_')[-1])
        return new_data_path
    def _make_label(self):
        label = []
        with open(os.path.join('utils', 'labels.yml')) as outfile:
            label_map = yaml.safe_load(outfile)
        oxford_labels = label_map['OXFORD_LABELS']
        for i in range(len(self.data_path)):
            class_name = '_'.join(self.data_path[i].split('.')[0].split('_')[:-1])
            label.append(oxford_labels[class_name])
        return label
    def __getitem__(self, index):
        if self.train:
            img = Image.open(os.path.join(self.root, 'images', self.train_path[index])).convert('RGB')
            label = self.train_label[index] 
        else:
            img = Image.open(os.path.join(self.root, 'images', self.test_path[index])).convert('RGB')
            label = self.test_label[index]
            
        if self.transforms:
            img = self.transforms(img)
        if self.train:   
            return self.train_path[index], img, label
        else:
            return self.test_path[index], img, label
    def __len__(self):
        if self.train:
            return len(self.train_path)
        else:
            return len(self.test_path) 

class Coreset_Dataset(data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = int(self.targets[index].item())
        
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
    
# https://github.com/wgrathwohl/JEM
def cycle(loader):
    while True:
        for data in loader:
            yield data

def get_dataset(args,
                root='./data', 
                dataset='splitted_mnist',  
                ):
    
    transform = get_transforms(args)
    if dataset == 'splitted_mnist':
        train_dataset = MNIST(root=root, 
                        train=True, 
                        transform=transform, 
                        download=True)
        
        test_dataset = MNIST(root=root, 
                        train=False, 
                        transform=transform, 
                        download=True)
        
    elif dataset == 'permuted_mnist':
        train_dataset = MNIST(root=root, 
                        train=True, 
                        transform=transform,
                        download=True)
        
        test_dataset = MNIST(root=root, 
                        train=False, 
                        transform=transform,
                        download=True)
        
    elif dataset == 'cifar10':
        train_dataset = CIFAR10(root=root, 
                          train=True, 
                          transform=transform, 
                          download=True)
        
        test_dataset = CIFAR10(root=root, 
                          train=False, 
                          transform=transform, 
                          download=True)
        
    elif dataset == 'cifar100':
        train_dataset = CIFAR100(root=root, 
                           train=True, 
                           transform=transform, 
                           download=True)
        test_dataset = CIFAR100(root=root, 
                           train=False, 
                           transform=transform, 
                           download=True)
    
    elif dataset == 'oxford_pet':
        train_dataset = Oxford_Pet(root=root,
                             data_path='images/',
                             transforms=transform,
                             train=True)
        test_dataset = Oxford_Pet(root=root,
                             data_path='images/',
                             transforms=transform,
                             train=False)
    
    else:
        raise NotImplementedError
    return train_dataset, test_dataset


def split_dataset(dataset, train_data, test_data, num_classes, num_tasks):
    num_remainders = num_classes % num_tasks
    if dataset == 'oxford_pet':
        train_data_splitted = [[] for _ in range(num_classes)]
        test_data_splitted = [[] for _ in range(num_classes)]
        print("=================Splitting Train Dataset=================")
        for i in tqdm(range(len(train_data) - num_remainders), colour='green', ncols=15, dynamic_ncols=True):
            train_data_splitted[train_data[i][2]].append(train_data[i])
        print("=================Splitting Test Dataset=================")
        for i in tqdm(range(len(test_data) - num_remainders)):
            test_data_splitted[test_data[i][2]].append(test_data[i], colour='green', ncols=15, dynamic_ncols=True)
        print(f"Excluded {num_remainders} classes")
        return train_data_splitted, test_data_splitted
    elif dataset == 'splitted_mnist' or dataset == 'cifar10' or dataset == 'cifar100':
        train_data_splitted = [[] for _ in range(num_classes)]
        test_data_splitted = [[] for _ in range(num_classes)]
        print("=================Splitting Train Dataset=================")
        for i in tqdm(range(len(train_data) - num_remainders), colour='green', ncols=15, dynamic_ncols=True):
            train_data_splitted[train_data[i][1]].append(train_data[i])
        print("=================Splitting Test Dataset=================")
        for i in tqdm(range(len(test_data) - num_remainders), colour='green', ncols=15, dynamic_ncols=True):
            test_data_splitted[test_data[i][1]].append(test_data[i])
        print(f"Excluded {num_remainders} classes")
        return train_data_splitted, test_data_splitted
    else:
        raise NotImplementedError

def make_tasks(dataset, img_size, num_channels, train_data_splitted, test_data_splitted, num_classes, num_tasks):
    if dataset == 'permuted_mnist':
        task_class_sets = []
        print("=================Making Tasks=================")
        for i, train_data in tqdm(enumerate(train_data_splitted), colour='green', ncols=15, dynamic_ncols=True):
            temp_task_class_sets=set()
            for j in range(len(train_data)):
                temp_task_class_sets.add(train_data[j][1] + 10 * i)
            task_class_sets.append(temp_task_class_sets)
        return train_data_splitted, test_data_splitted, task_class_sets
    
    elif dataset == 'oxford_pet':
        train_tasks = []
        test_tasks = []
        task_class_sets = []
        
        print("=================Combining Classes=================")
        for i in tqdm(range(num_classes), colour='green', ncols=15, dynamic_ncols=True):
            train_list, test_list = [], []
            for id, img, label in train_data_splitted[i]:
                train_list.append((id, img.view(img_size, img_size, num_channels).reshape(num_channels, img_size, img_size), label))
            for id, img, label in test_data_splitted[i]:
                test_list.append((id, img.view(img_size, img_size, num_channels).reshape(num_channels, img_size, img_size), label))

            locals()['train_'+str(i)+'_3d'] = train_list
            locals()['test_'+str(i)+'_3d'] = test_list
            
        class_counter = 0
        print("=================Making Tasks=================")
        print(f"Num of Tasks : {num_tasks}")
        print(f"Num of class : {num_classes}")
        for i in tqdm(range(num_tasks), colour='green', ncols=15, dynamic_ncols=True):
            temp_train_task = []
            temp_test_task = []
            temp_task_class_sets = set()
            for j in range(num_classes//num_tasks):
                temp_train_task = temp_train_task + locals()['train_'+str(i+class_counter+j)+'_3d']
                temp_test_task = temp_test_task + locals()['test_'+str(i+class_counter+j)+'_3d']
                temp_task_class_sets.add(i+class_counter+j)
            train_tasks.append(temp_train_task)
            test_tasks.append(temp_test_task)
            task_class_sets.append(temp_task_class_sets)
            class_counter+=(num_classes//num_tasks - 1)
        return train_tasks, test_tasks, task_class_sets
    
    elif dataset == 'splitted_mnist' or dataset == 'cifar10' or dataset == 'cifar100':
        train_tasks = []
        test_tasks = []
        task_class_sets = []
        
        print("=================Combining Classes=================")
        for i in tqdm(range(num_classes), colour='green', ncols=15, dynamic_ncols=True):
            train_list, test_list = [], []
            for img, label in train_data_splitted[i]:
                train_list.append((img.view(img_size, img_size, num_channels).reshape(num_channels, img_size, img_size), label))
            for img, label in test_data_splitted[i]:
                test_list.append((img.view(img_size, img_size, num_channels).reshape(num_channels, img_size, img_size), label))

            locals()['train_'+str(i)+'_3d'] = train_list
            locals()['test_'+str(i)+'_3d'] = test_list
            
        class_counter = 0
        print("=================Making Tasks=================")
        print(f"Num of Tasks : {num_tasks}")
        print(f"Num of class : {num_classes}")
        for i in tqdm(range(num_tasks), colour='green', ncols=15, dynamic_ncols=True):
            temp_train_task = []
            temp_test_task = []
            temp_task_class_sets = set()
            for j in range(num_classes//num_tasks):
                temp_train_task = temp_train_task + locals()['train_'+str(i+class_counter+j)+'_3d']
                temp_test_task = temp_test_task + locals()['test_'+str(i+class_counter+j)+'_3d']
                temp_task_class_sets.add(i+class_counter+j)
            train_tasks.append(temp_train_task)
            test_tasks.append(temp_test_task)
            task_class_sets.append(temp_task_class_sets)
            class_counter+=(num_classes//num_tasks - 1)
        return train_tasks, test_tasks, task_class_sets
    
    else:
        raise NotImplementedError

def get_loader(args, train_tasks, test_tasks):
    train_loaders, test_loaders = [], []
    print("=================Making Train Dataloader=================")
    for train_task in tqdm(train_tasks, colour='green', ncols=15, dynamic_ncols=True):
        train_loaders.append(DataLoader(train_task, 
                                        args.batch_size, 
                                        shuffle=True, 
                                        num_workers=args.num_workers, 
                                        pin_memory=args.cuda, 
                                        drop_last=True))
    print("=================Making Test Dataloader=================")
    for test_task in tqdm(test_tasks, colour='green', ncols=15, dynamic_ncols=True):
        test_loaders.append(DataLoader(test_task, 
                                       args.test_batch_size, 
                                       shuffle=False, 
                                       num_workers=args.num_workers, 
                                       pin_memory=args.cuda, 
                                       drop_last=False))
    return train_loaders, test_loaders

def prepare_data(args):
    if args.dataset == 'permuted_mnist':
        train_datasets, test_datasets = [get_dataset(args,
                                                     root=args.data_root,
                                                     dataset=args.dataset) for i in range(args.num_tasks)]
        train_tasks, test_tasks, task_class_sets = make_tasks(dataset=args.dataset,
                                                              img_size=args.img_size,
                                                              num_channels=args.num_channels,
                                                              train_data_splitted=train_datasets,
                                                              test_data_splitted=test_datasets,
                                                              num_classes=args.num_classes,
                                                              num_tasks=args.num_tasks)
        train_loaders, test_loaders = get_loader(args=args,
                                                 train_tasks=train_tasks,
                                                 test_tasks=test_tasks)
        
    else:
        train_dataset, test_dataset = get_dataset(args,
                                                  root=args.data_root,
                                                  dataset=args.dataset)
        splitted_train_dataset, splitted_test_dataset = split_dataset(dataset=args.dataset, 
                                                                      train_data=train_dataset, 
                                                                      test_data=test_dataset, 
                                                                      num_classes=args.num_classes,
                                                                      num_tasks=args.num_tasks)
        train_tasks, test_tasks, task_class_sets = make_tasks(dataset=args.dataset,
                                                              img_size=args.img_size, 
                                                              num_channels=args.num_channels,
                                                              train_data_splitted=splitted_train_dataset,
                                                              test_data_splitted=splitted_test_dataset,
                                                              num_classes=args.num_classes,
                                                              num_tasks=args.num_tasks)
        train_loaders, test_loaders = get_loader(args=args, 
                                                 train_tasks=train_tasks,
                                                 test_tasks=test_tasks)
    return train_loaders, test_loaders, task_class_sets