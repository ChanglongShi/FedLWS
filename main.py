from datasets import Data
from nodes import Node
from args import args_parser
from utils import *
from server_funct import *
from client_funct import *
import os
import time
import pandas as pd
# import numpy as np
from collections import Counter

 



if __name__ == '__main__':

    args = args_parser()

    # Set random seeds
    setup_seed(args.random_seed)
    print(args)

    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device


    # Loading data
    data = Data(args)

    
    sample_size = []
    for i in range(args.node_num): 
        sample_size.append(len(data.train_loader[i]))
    size_weights = [i/sum(sample_size) for i in sample_size]
    central_node = Node(-1, data.test_loader[0], data.test_set, args)


    # Initialize the client nodes
    client_nodes = {}
    ndata=[]
    for i in range(args.node_num): 
        client_nodes[i] = Node(i, data.train_loader[i], data.train_set, args) 
    # best_test_acc_recorder = Best_auc()
    test_acc_recorder = []
    avgtime=[]
    for rounds in range(args.T):
        torch.cuda.empty_cache()
        print('===============Stage 1 The {:d}-th round==============='.format(rounds + 1))
        lr_scheduler(rounds, client_nodes, args)
        # Client update
        if args.client_method == 'feddisco':
            client_nodes, train_loss,dks,outs = Client_update(args, client_nodes, central_node)
        else:
            client_nodes, train_loss = Client_update(args, client_nodes, central_node)
        avg_client_acc = Client_validate(args, client_nodes)
        print(args.server_method + args.client_method + ', averaged clients personalization acc is ', avg_client_acc)
        
        # Partial select function
        if args.select_ratio == 1.0:
            select_list = [idx for idx in range(len(client_nodes))]
        else:
            select_list = generate_selectlist(client_nodes, args.select_ratio)
       


        # Server update
        start = time.perf_counter() 
        
        central_node = Server_update(args, central_node, client_nodes, select_list, size_weights)
        
        end = time.perf_counter() 


        acc = validate(args, central_node, which_dataset = 'local')
        print(args.server_method + args.client_method + ', global model test acc is ', acc)
        test_acc_recorder.append(acc)

        print('Running time: %s Seconds' % (end - start))
        # Final acc recorder
        
        # best_test_acc_recorder.update(acc)
        avgtime.append(end - start)
        best_acc = max(test_acc_recorder)
        print("Current_Best test acc is:", best_acc)
        
    
   
    print("Avg runing time:",np.mean(avgtime))

    # best acc
    best_acc = max(test_acc_recorder)
    print("Final_Best test acc is:", best_acc)

    # best top 10
    top_10_acc = sorted(test_acc_recorder, reverse=True)[:10]
    top_10_avg = np.mean(top_10_acc)
    top_10_std = np.std(top_10_acc)
    print("Top 10 test acc avg is:", top_10_avg)
    print("Top 10 test acc std is:", top_10_std)