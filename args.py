import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--noniid_type', type=str, default='dirichlet',
                        help="iid or dirichlet")
    parser.add_argument('--iid', type=int, default=0,  
                        help='set 1 for iid')
    parser.add_argument('--batchsize', type=int, default=128, 
                        help="batchsize")
    parser.add_argument('--validate_batchsize', type=int, default=128, 
                        help="batchsize")
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5, 
                    help="dirichlet_alpha")
    parser.add_argument('--dirichlet_alpha2', type=float, default=False, 
                    help="dirichlet_alpha2")
    

    # System
    parser.add_argument('--device', type=str, default='1',
                        help="device: {cuda, cpu}")
    parser.add_argument('--node_num', type=int, default=20, # 200
                        help="Number of nodes")
    parser.add_argument('--T', type=int, default=200,  # 100 
                        help="Number of communication rounds")
    parser.add_argument('--E', type=int, default=1, # 3
                        help="Number of local epochs: E")
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help="Type of algorithms:{cifar10,cifar100, fmnist,tinyimagenet}") 
    parser.add_argument('--select_ratio', type=float, default=1.0,
                    help="the ratio of client selection in each round")
    parser.add_argument('--local_model', type=str, default='ResNet20',
                        help='Type of local model: {CNN, ResNet20, ResNet18, WRN56_4}')
    parser.add_argument('--random_seed', type=int, default=10,
                        help="random seed for the whole experiment")
    parser.add_argument('--min_tau', type=float, default=0.01,
                        help="min of tau")
    parser.add_argument('--max_tau', type=float, default=0.2,
                        help="max of tau")
    parser.add_argument('--beta', type=float, default=0.03,
                        help="hyper parameter")
    parser.add_argument('--longtail_clients', type=str, default="none",)
    


    # Server function
    parser.add_argument('--server_method', type=str, default='fedlws',
                        help="fedavg, fedlws")
    parser.add_argument('--server_valid_ratio', type=float, default=0.02, 
                    help="the ratio of proxy dataset in the central server")
    parser.add_argument('--a', type=int, default=0.1,
                        help="balance term for feddisco")
    parser.add_argument('--b', type=int, default=0.1,
                        help="adjust term for feddisco")
                        
    # Client function
    parser.add_argument('--client_method', type=str, default='local_train',
                        help="local_train, fedprox, feddyn,feddisco")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="optimizer: {sgd, adam}")
    parser.add_argument('--client_valid_ratio', type=float, default=0.3,
                    help="the ratio of validate set in the clients")  
    parser.add_argument('--lr', type=float, default=0.08,
                        help='clients loca learning rate')
    parser.add_argument('--local_wd_rate', type=float, default=5e-4,
                        help='clients local wd rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='clients SGD momentum')
    parser.add_argument('--mu', type=float, default=0.001,
                        help="clients proximal term mu for FedProx")

    args = parser.parse_args()

    return args
