import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import copy

from torchvision.datasets import ImageFolder, DatasetFolder


# https://github.com/QinbinLi/MOON/blob/6c7a4ed1b1a8c0724fa2976292a667a828e3ff5d/datasets.py#L148
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)
        

# Subset function
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]] 
        return image, label
    
# Main data loader
class Data(object):
    def __init__(self, args):
        self.args = args
        node_num = args.node_num
        if args.dataset == 'cifar10':
            # Data enhancement: None
            tra_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            val_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

            self.train_set = torchvision.datasets.CIFAR10(
                root="./Dataset/cifar/", train=True, download=True, transform=tra_transformer
            )
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)
                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(random_state=random_state, dataset=self.train_set, non_iid_alpha1=args.dirichlet_alpha,non_iid_alpha2=args.dirichlet_alpha2 ,num_classes=10, num_indices=num_indices, n_workers=node_num)  
                elif args.longtail_clients != 'none':
                    groups, proportion = build_non_iid_by_dirichlet_LT(random_state=random_state, dataset=self.train_set, lt_rho=args.longtail_clients, non_iid_alpha=args.dirichlet_alpha, num_classes=10, num_indices=num_indices, n_workers=node_num)  
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(random_state=random_state, dataset=self.train_set, non_iid_alpha=args.dirichlet_alpha, num_classes=10, num_indices=num_indices, n_workers=node_num)  
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion
                
                # print(sum(proportion))
                # print(proportion.shape)
                # exit()
            else:
                data_num = [int(50000/node_num) for _ in range(node_num)]
                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.test_set = torchvision.datasets.CIFAR10(
                root="./Dataset/cifar/", train=False, download=True, transform=val_transformer
            )
                # exit()


            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])
            # print(self.test_loader)
            # print(self.test_loader[0])
            # print(self.test_loader[0].indices)
            # exit()


        elif args.dataset == 'cifar100':
            # Data enhancement
            tra_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            val_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

            self.train_set = torchvision.datasets.CIFAR100(
                root="./Dataset/cifar/", train=True, download=True, transform=tra_transformer
            )
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)
                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(random_state=random_state, dataset=self.train_set, non_iid_alpha1=args.dirichlet_alpha,non_iid_alpha2=args.dirichlet_alpha2 ,num_classes=100, num_indices=num_indices, n_workers=node_num)  
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(random_state=random_state, dataset=self.train_set, non_iid_alpha=args.dirichlet_alpha, num_classes=100, num_indices=num_indices, n_workers=node_num)  
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion
            else:
                data_num = [int(50000/node_num) for _ in range(node_num)]
                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.test_set = torchvision.datasets.CIFAR100(
                root="./Dataset/cifar/", train=False, download=True, transform=val_transformer
            )
            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])

        elif args.dataset == 'fmnist':
            # Data enhancement
            tra_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
            val_transformer = transforms.Compose(
                [
                    transforms.ToTensor()
                ]
            )
            self.train_set = torchvision.datasets.FashionMNIST(
                root="./Dataset/FashionMNIST", train=True, download=True, transform=tra_transformer
            )
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)
                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(random_state=random_state, dataset=self.train_set, non_iid_alpha1=args.dirichlet_alpha,non_iid_alpha2=args.dirichlet_alpha2 ,num_classes=100, num_indices=num_indices, n_workers=node_num)  
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(random_state=random_state, dataset=self.train_set, non_iid_alpha=args.dirichlet_alpha, num_classes=100, num_indices=num_indices, n_workers=node_num)  
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion
            else:
                data_num = [int(60000/node_num) for _ in range(node_num)]
                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.test_set = torchvision.datasets.FashionMNIST(
                root="./Dataset/FashionMNIST", train=False, download=True, transform=val_transformer
            )
            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])

        

        elif args.dataset == 'tinyimagenet':
            # Data enhancement
            tra_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            val_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

            self.train_set = torchvision.datasets.ImageFolder(
                root='./Dataset/tiny-imagenet-200/train/',  transform=tra_transformer
            )
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)
                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(random_state=random_state, dataset=self.train_set, non_iid_alpha1=args.dirichlet_alpha,non_iid_alpha2=args.dirichlet_alpha2 ,num_classes=200, num_indices=num_indices, n_workers=node_num)  
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(random_state=random_state, dataset=self.train_set, non_iid_alpha=args.dirichlet_alpha, num_classes=200, num_indices=num_indices, n_workers=node_num)  
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion
            else:
                data_num = [int(100000/node_num) for _ in range(node_num)]
                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.test_set = torchvision.datasets.ImageFolder(
                root="./Dataset/tiny-imagenet-200/val/", transform=val_transformer
            )
            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])

        


### Dirichlet noniid functions ###
def build_non_iid_by_dirichlet_hybrid(
    random_state = np.random.RandomState(0), dataset = 0, non_iid_alpha1 = 10, non_iid_alpha2 = 1, num_classes = 10, num_indices = 60000, n_workers = 10
):

    indicesbyclass = {}
    for i in range(num_classes):
        indicesbyclass[i] = []
    
    for idx, target in enumerate(dataset.targets):
        indicesbyclass[int(target)].append(idx)
    
    for i in range(num_classes):
        random_state.shuffle(indicesbyclass[i])
    
    partition = random_state.dirichlet(np.repeat(non_iid_alpha1, n_workers), num_classes).transpose()

    partition2 = random_state.dirichlet(np.repeat(non_iid_alpha2, n_workers/2), num_classes).transpose()

    new_partition1 = copy.deepcopy(partition[:int(n_workers/2)])

    sum_distr1 = np.sum(new_partition1, axis=0)

    diag_mat = np.diag(1 - sum_distr1)

    new_partition2 = np.dot(diag_mat, partition2.T).T

    client_partition = np.vstack((new_partition1, new_partition2))

    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            client_partition[i][j] = int(round(client_partition[i][j]*len(indicesbyclass[j])))
    
    client_partition_index = copy.deepcopy(client_partition)
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                client_partition_index[i][j] = client_partition_index[i][j]
            elif i == len(client_partition) - 1:
                client_partition_index[i][j] = len(indicesbyclass[j])
            else:
                client_partition_index[i][j] = client_partition_index[i-1][j] + client_partition_index[i][j]
            
    dict_users = {}
    for i in range(n_workers):
        dict_users[i] = []
    
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                dict_users[i].extend(indicesbyclass[j][:int(client_partition_index[i][j])])
            else:
                dict_users[i].extend(indicesbyclass[j][int(client_partition_index[i-1][j]) : int(client_partition_index[i][j])])
    
    for i in range(len(dict_users)):
        random_state.shuffle(dict_users[i])

    return dict_users, client_partition

def build_non_iid_by_dirichlet_new(
    random_state = np.random.RandomState(0), dataset = 0, non_iid_alpha = 10, num_classes = 10, num_indices = 60000, n_workers = 10
):

    indicesbyclass = {}
    for i in range(num_classes):
        indicesbyclass[i] = []
    # print(dataset.targets)
    # exit()
    for idx, target in enumerate(dataset.targets):
        
        indicesbyclass[int(target)].append(idx)
    
    for i in range(num_classes):
        random_state.shuffle(indicesbyclass[i])
    
    client_partition = random_state.dirichlet(np.repeat(non_iid_alpha, n_workers), num_classes).transpose()
    # print(client_partition.shape)
    # exit()
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            client_partition[i][j] = int(round(client_partition[i][j]*len(indicesbyclass[j])))
    
    client_partition_index = copy.deepcopy(client_partition)
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                client_partition_index[i][j] = client_partition_index[i][j]
            elif i == len(client_partition) - 1:
                client_partition_index[i][j] = len(indicesbyclass[j])
            else:
                client_partition_index[i][j] = client_partition_index[i-1][j] + client_partition_index[i][j]
    # print(client_partition_index.shape)
    # exit()
    dict_users = {}
    for i in range(n_workers):
        dict_users[i] = []
    
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                dict_users[i].extend(indicesbyclass[j][:int(client_partition_index[i][j])])
            else:
                dict_users[i].extend(indicesbyclass[j][int(client_partition_index[i-1][j]) : int(client_partition_index[i][j])])
    

    # print(dict_users[0])
    # exit()
    for i in range(len(dict_users)):
        random_state.shuffle(dict_users[i])
    # print(dict_users[0].indices)
    # exit()
    return dict_users, client_partition


def build_non_iid_by_dirichlet_LT(
    random_state = np.random.RandomState(0), dataset = 0,  lt_rho = 10.0, non_iid_alpha = 10, num_classes = 10, num_indices = 60000, n_workers = 10
):
    # generate indicesbyclass list
    indicesbyclass = {}
    for i in range(num_classes):
        indicesbyclass[i] = []
    for idx, target in enumerate(dataset.targets):
        indicesbyclass[int(target)].append(idx)

    # calculate the image per class for LT
    # reformulate the indicesbyclass according to the image per class
    imb_factor = 1/float(lt_rho)
    for _classes_idx in range(num_classes):
        num = int(len(indicesbyclass[_classes_idx]) * (imb_factor**(_classes_idx / (num_classes - 1.0))))
        random_state.shuffle(indicesbyclass[_classes_idx])
        indicesbyclass[_classes_idx] = indicesbyclass[_classes_idx][:num]
    
    client_partition = random_state.dirichlet(np.repeat(non_iid_alpha, n_workers), num_classes).transpose()

    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            client_partition[i][j] = int(round(client_partition[i][j]*len(indicesbyclass[j])))
    
    client_partition_index = copy.deepcopy(client_partition)
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                client_partition_index[i][j] = client_partition_index[i][j]
            elif i == len(client_partition) - 1:
                client_partition_index[i][j] = len(indicesbyclass[j])
            else:
                client_partition_index[i][j] = client_partition_index[i-1][j] + client_partition_index[i][j]
            
    dict_users = {}
    for i in range(n_workers):
        dict_users[i] = []
    
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                dict_users[i].extend(indicesbyclass[j][:int(client_partition_index[i][j])])
            else:
                dict_users[i].extend(indicesbyclass[j][int(client_partition_index[i-1][j]) : int(client_partition_index[i][j])])
    
    for i in range(len(dict_users)):
        random_state.shuffle(dict_users[i])

    return dict_users, client_partition
