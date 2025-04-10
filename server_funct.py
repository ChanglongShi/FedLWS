import numpy as np
import torch
import torch.nn.functional as F
import math
import torch.optim as optim
import torch.nn as nn
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import init_model
import math
from copy import deepcopy
import warnings
import torch
from torch.nn import Module
from sklearn.metrics.pairwise import cosine_similarity
import datetime
##############################################################################
# General server function
##############################################################################

def receive_client_models(args, client_nodes, select_list, size_weights):
    client_params = []
    for idx in select_list:
        client_params.append(copy.deepcopy(client_nodes[idx].model.state_dict()))
    
    agg_weights = [size_weights[idx] for idx in select_list]
    agg_weights = [w/sum(agg_weights) for w in agg_weights]
    # if

    return agg_weights, client_params

def get_model_updates(client_params, prev_para):
    prev_param = copy.deepcopy(prev_para)
    client_updates = []
    for param in client_params:
        client_updates.append(param.sub(prev_param))
    return client_updates

def get_client_params_with_serverlr(server_lr, prev_param, client_updates):
    client_params = []
    with torch.no_grad():
        for update in client_updates:
            param = prev_param.add(update*server_lr)
            client_params.append(param)
    return client_params


def Server_update(args, central_node, client_nodes, select_list, size_weights,dks=None,outs=None):
    '''
    server update functions for baselines
    '''

    # receive the local models from clients
    agg_weights, client_params = receive_client_models(args, client_nodes, select_list, size_weights)

    # update the global model
    if args.server_method == 'fedavg':
        avg_global_param = fedavg(args,client_params, agg_weights)
        central_node.model.load_state_dict(avg_global_param)
  
    elif args.server_method == 'fedlws':
        
        avg_global_param=fedlws(args,client_params,central_node,agg_weights,outs)
        central_node.model.load_state_dict(avg_global_param)
    

    

    else:
        raise ValueError('Undefined server method...')

   
    return central_node






# Fedlws 
def fedlws(args,parameters, central_node,list_nums_local_data, dks=None,outs=None):
   
    param=central_node.model.state_dict()
    global_params = copy.deepcopy(param)
    fedavg_global_params = copy.deepcopy(parameters[0])
    



    for name_param in parameters[0]:
        list_values_param = []

        for dict_local_params, num_local_data in zip(parameters,list_nums_local_data):
            list_values_param.append(dict_local_params[name_param] * num_local_data)
        value_global_param = sum(list_values_param)# / sum(list_nums_local_data)
        fedavg_global_params[name_param] = value_global_param


    if args.local_model=='Vit':

        cur_w=[]
        last_w=[]
        l=torch.tensor([]).cuda()
        l_last=torch.tensor([]).cuda()
        for name_param in parameters[0]:
            l=torch.cat((l,fedavg_global_params[name_param].reshape(-1)))
            l_last=torch.cat((l_last,global_params[name_param].reshape(-1)))
            if ("1.fn.net.3.bias" in name_param) or (name_param=="mlp_head.1.bias") or (name_param =="mlp_head.0.bias"):
                cur_w.append(l)
                last_w.append(l_last)
                l=torch.tensor([]).cuda()
                l_last=torch.tensor([]).cuda()
        clients_w=[]
        for i in range(len(parameters)):
            client_w=[]
            l_client=torch.tensor([]).cuda()
            for name_param in parameters[0]:
                l_client=torch.cat((l_client,parameters[i][name_param].reshape(-1)))
                # a=layer_cossim[i][name_param]
                if ("1.fn.net.3.bias" in name_param) or (name_param=="mlp_head.1.bias") or (name_param =="mlp_head.0.bias"):
                    client_w.append(l_client)
                    # a=torch.tensor(0).cuda().float()
                    l_client=torch.tensor([]).cuda()
            clients_w.append(client_w)
            
    else:

        cur_w=[]
        last_w=[]
        l=torch.tensor([]).cuda()
        l_last=torch.tensor([]).cuda()
        for name_param in parameters[0]:
            # print(name_param)
            l=torch.cat((l,fedavg_global_params[name_param].reshape(-1)))
            l_last=torch.cat((l_last,global_params[name_param].reshape(-1)))
            # a=layer_cossim[i][name_param]
            if "bias" in name_param:
                cur_w.append(l)
                last_w.append(l_last)
                # print(name_param)
                # print(l.shape)
                # a=torch.tensor(0).cuda().float()
                l=torch.tensor([]).cuda()
                l_last=torch.tensor([]).cuda()
        clients_w=[]
        for i in range(len(parameters)):
            client_w=[]
            l_client=torch.tensor([]).cuda()
            for name_param in parameters[0]:
                l_client=torch.cat((l_client,parameters[i][name_param].reshape(-1)))
                # a=layer_cossim[i][name_param]
                if "bias" in name_param:
                    client_w.append(l_client)
                    # a=torch.tensor(0).cuda().float()
                    l_client=torch.tensor([]).cuda()
            clients_w.append(client_w)

    layer_gammas=[]

  
    ######### layer_tau ##############3
    
    taus=[]
    for i in range(len(last_w)):
        # print(cur_w[i].shape)
        grad=torch.norm(cur_w[i]-last_w[i],p=2)
        layer_grad=[]
        for k in range(len(parameters)):
            layer_grad.append(clients_w[k][i]-last_w[i])
        global_grad=torch.mean(torch.stack(layer_grad),dim=0)
        l2_norms = [torch.norm(tensor - global_grad, p=2) for tensor in layer_grad]
        l2_norm_average = sum(l2_norms) / len(l2_norms)
        tau=args.beta*(l2_norm_average)
        tau = torch.clamp(tau, min=args.min_tau, max=args.max_tau)
        taus.append(tau)
        gamma=torch.norm(last_w[i],p=2)/(torch.norm(last_w[i],p=2)+tau*grad)
        layer_gammas.append(gamma)

    print("taus:",taus)



    print("layer_gammas:",layer_gammas)

    cur_layer=0
    for name_param in parameters[0]:
        # if name_param[-6:]=="weight":
        #     k+=1
                  
        fedavg_global_params[name_param] = fedavg_global_params[name_param]*layer_gammas[cur_layer]#+fedavg_global_params[name_param]*(1-d0)
        if args.local_model=='Vit':
            if ("1.fn.net.3.bias" in name_param) or (name_param=="mlp_head.1.bias" or (name_param =="mlp_head.0.bias")):
                cur_layer+=1
        else:
            if "bias" in name_param:
                cur_layer+=1

   
    return fedavg_global_params


# FedAvg
def fedavg(args,parameters, list_nums_local_data):
    fedavg_global_params = copy.deepcopy(parameters[0])
    # d=[]
    for name_param in parameters[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(parameters, list_nums_local_data):
            list_values_param.append(dict_local_params[name_param] * num_local_data)
        value_global_param = sum(list_values_param) / sum(list_nums_local_data)
        fedavg_global_params[name_param] = value_global_param
    
    return fedavg_global_params




