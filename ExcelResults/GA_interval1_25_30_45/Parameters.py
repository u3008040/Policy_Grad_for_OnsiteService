# -*- coding: utf-8 -*-
import argparse

def parameters():
    #RL parameters
    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--poissonlambda', default=10, type=int, help='order arrival rate follows a poisson distribution')
    parser.add_argument('--hidden_dim',default=128,type=int,help='hidden dimensions')
    parser.add_argument('--gamma',default=1,type=int,help='reward discount ratio')
    parser.add_argument('--postman_dim', default=12, type=int, help='Number of features needed for a postman')
    parser.add_argument('--node_dim', default=16, type=int,
                        help='dimension of nodal state')  # number of state information for each node (customer)
    parser.add_argument('--voc_edges_in', default=4, type=int, help='number of edges, not edge state number')
    parser.add_argument('--num_layers', default=2, type=int, help='Number of iterations in neural network layers')
    parser.add_argument('--edge_dim', default=1, type=int, help='dimension of edge state information')
    parser.add_argument('--aggregation', default='mean', type=str, help='sum or mean')
    parser.add_argument('--learningrate',default=7e-4,type=float,help='')
    parser.add_argument('--solving_batch_size', default=50)# 测试时候的batch
    parser.add_argument('--training_batch_size', default=12)# 训练时候的batch
    parser.add_argument('--training_size',default=2) # howmany batches per training
    parser.add_argument('--eps_clip',default=0.2,help='max difference between old and new policy')##@?
    parser.add_argument('--buffersize', default=400)
    parser.add_argument('--trainingepisode', default=400)
    parser.add_argument('--gap',default=5,help='how long it takes to reoptimise')
    parser.add_argument('--baseline_update_steps',default=5,help='how often baseline is updated')
    parser.add_argument('--exploration_prob', default=0.2, help='Probability of exploration')
    
    #environment parameters
    parser.add_argument('--Reject_NA_rate', default=0, type=float, help='probability of rejecting no assignment')
    parser.add_argument('--Greedyprocessratio',default=0.05,type=float,help='probability of greedy process')
    parser.add_argument('--Distancepenalty',default=0,type=float,help='distance penalty')
    parser.add_argument('--entropyrate',default=1,type=float,help='entropy rate')
    parser.add_argument('--entropystepdecay',default=0.1*1/200,type=float,help='entropy step decay')
    parser.add_argument('--speed',default=0.1,type=float,help='Speed of the postman')
    parser.add_argument('--unittime',default=1,type=float,help='time interval')
    parser.add_argument('--totaltime', default=2000, type=float, help='total time interval')
    parser.add_argument('--solvingepisode',default=100)
    parser.add_argument('--low_order_bound', default=25, type=int,help='lowest number of orders')
    parser.add_argument('--high_order_bound',default=30,type=int)
    parser.add_argument('--low_stay_bound',default=1,type=int,help='how long postman stays')
    parser.add_argument('--high_stay_bound',default=4,type=int)
    parser.add_argument('--TWlb',default=20,type=int)
    parser.add_argument('--TWub',default=40,type=int)
    parser.add_argument('--INnumber', default=80, type=int)
    parser.add_argument('--CPnumber', default=10, type=int)
    parser.add_argument('--NNlower',default=8,type=int,help='Number of nodes in a map')
    parser.add_argument('--NNhigher',default=8,type=int,help='Number of nodes in a map upper bounds')
    parser.add_argument('--PNlower',default=3,type=int,help='Number of postmen in a map lower bound')
    parser.add_argument('--PNhigher',default=3,type=int,help='Number of postmen in a map higher bound')
    parser.add_argument('--predict_time_grain',default=6,type=int,help='predict the order number within predict_time_grain unittime')
    args = parser.parse_args([])
    return args


