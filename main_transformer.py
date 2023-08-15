# -*- coding: utf-8 -*-
import sys, torch, copy, time, os, time
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import copy
import math

#--- own packages
import utils.Parameters as Parameters
import Environment as Env
from algorithm.baseline.GreedyOptimisation import Greedyoptimiser
from more_itertools import sort_together
from utils.my_utils import dir_check, flatten, write_xlsx, multi_thread_work, Exp_Buffer, is_all_none
from pprint import pprint

args = Parameters.parameters()
args = vars(args)


# ---------------control panel------------------------
is_training = 1
multithread=0
GreedyRolloutBaseline=0
nodispatch=True
Euclidean=0
Mag_Factor=1  #how many times this graph is magnified

softupdate=1
tau=0.08
display_myopic = True
PPO = False
display_RL = 1
display_VRPTW = False
is_predict = False

if_display_details = 0
serverversion = False
graph_seed = 0
#customerseed = 5
if_display_elapse=0
#if nodispatch:
#    import NetModel_nodispatch as NetModel
#else:
#    import NetModel
import algorithm.gcn_dispatch.NetModel_transformer as NetModel
local_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
path = f'./output/{local_time}/'
path_actor = './output/'
dir_check(path)
actordict = path_actor + 'actorreal1.pth'  # 神经网络参数
if display_VRPTW:
    import VRPTW
# -----------------hyperparameters--------------------
if display_myopic == False and display_RL == False:
    print('wrong, both displays are disabled')
    sys.exit()
global actor_net,Reject_NA_Rate,Reject_decay
global buffer
global totaltime, speed, buffersize
NNlower = args['NNlower']
NNhigher = args['NNhigher']
lam = args['lam']
speed = args['speed']
PNlower = args['PNlower']
PNhigher = args['PNhigher']
learningrate = args['learningrate']
lb = args['low_order_bound']
hb = args['high_order_bound']
entropystepdecay = args['entropystepdecay']
totaltime = args['totaltime']
training_size = args['training_size']
postman_dim = args['postman_dim']
node_dim = args['node_dim']
Reject_NA_Rate=args['Reject_NA_rate']
baseline_update_steps = args['baseline_update_steps']

if is_training == True:
    num_episodes = args['trainingepisode']
    batch_size = args['training_batch_size']
else:
    num_episodes = args['solvingepisode']
    batch_size = args['solving_batch_size']

buffersize = num_episodes
samplesize = batch_size
gamma = args['gamma']
global buffer
buffer = Exp_Buffer(batch_size*training_size)

Reject_decay=0.1*(1/200)

# -----------------------------------set device to CPU or GPU-----------------------------------
global actor_net
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

actor_net = NetModel.ResidualGatedGCNModel(is_training, PNlower, Mag_Factor)
actor_net.to(device)

def RLpolicy(g, PN, state, batch, iteration,Reject_NA_Rate, initial=False,
             neuralnetwork=None, istraining=is_training,NN=0,device='cuda',Greedyprocess=False):  # training indicates if training or playing experience
    if if_display_elapse: print('entrance elapse',state.elapse)
    if if_display_details:
        print('current time', state.current_time)
        print('len of order_popup_time', len(state.order_popuptime), 'len of order_node', len(state.order_node),
              'len of unassigned_order_node',
              len(state.unassigned_order_node), 'len of unassigned_order_popuptime',
              len(state.unassigned_order_popuptime))
        print('postman_destination_node:', state.postman_destination_node)
    xcoords, ycoords = g.get_vertices()
    postmanx, postmany, postman_status, postman_percentage = g.get_postman()
    if initial:
        # init state information
        for i in range(len(state.order_popuptime)):
            if state.order_popuptime[i] <= 0:  # 主要是看等于0
                state.unassigned_order_node.append(state.order_node[i])
                state.unassigned_order_indice.append(state.order_indice[i])
                state.unassigned_order_popuptime.append(state.order_popuptime[i])
                state.unassigned_order_staytime.append(state.order_stay_time[i])
                state.node_earliest_popuptime[int(state.order_node[i])] = state.order_popuptime[i]
                state.node_latest_popuptime[int(state.order_node[i])] = state.order_popuptime[i]
                state.unassigned_order_timewindow.append(state.order_timewindow[i])
                g.add_order_node(state.order_node[i])
                g.node_all_orders[state.unassigned_order_node[i]] += 1
            if state.order_popuptime[i]>0:
                state.elapse=state.order_popuptime[i]
                break

        if if_display_elapse: print('initial elapse',state.elapse)
        # init graph information
        for i in range(PN):
            g.update_postman(xcoords[int(state.postman_prev_node[i])], ycoords[int(state.postman_prev_node[i])], i, 0,
                             0)
        if state.postman_destination_node == False or state.postman_destination_node == [] or state.postman_destination_node == None:
            state.postman_destination_node = [None for _ in range(PN)]
            state.postman_destination_indice = [None for _ in range(PN)]
        state.change_elapse = [False for _ in range(len(state.unassigned_order_node))]
    else:
        for i in range(len(state.order_popuptime)):
            if len(state.order_popuptime) == 0: break
            if state.order_popuptime[i] - state.current_time > 0:
                if if_display_elapse: print('elapse assigned by next arriving order', state.elapse,
                      state.order_popuptime[i] - state.current_time)
                state.elapse = min(state.elapse, state.order_popuptime[i] - state.current_time)
                break
        if state.elapse == totaltime:
            if len(state.unassigned_order_node) == 0 and len(state.order_node) == 0 and is_all_none(state.postman_destination_node):  # all assignments are made
                state.Done = True
                if if_display_details: print('exit1')
                state.current_time = state.current_time + state.elapse
                return g, state
            elif len(state.unassigned_order_node) != 0:  # there are still remaining tasks at the same location the postman stays
                state.elapse = min(state.unassigned_order_staytime)
                if if_display_elapse: print('elapse assigned by unassigned',min(state.unassigned_order_staytime))
        if if_display_details:
            print('current time', state.current_time, 'elapsed', state.elapse)
            print('current_order_node', g.current_order_node)
        # need to add another thing here, stating that if further elapse, then the first one in the list should
        # be assigned a postman then evaluate the shortest travel duration, then set the elapse time.
        state.change_elapse = [False for _ in range(len(state.unassigned_order_node))]  # this is for individual postman assignment,
        # not for everyone. will not triger recursion
        for i in range(len(state.order_node)):  # find all orders that lie inside this elapse time interval.
            if state.current_time >= state.order_popuptime[i] > state.previous_time:
                state.unassigned_order_node = state.unassigned_order_node + [int(state.order_node[i])]
                state.unassigned_order_indice = state.unassigned_order_indice + [int(state.order_indice[i])]
                state.unassigned_order_popuptime = state.unassigned_order_popuptime + [state.order_popuptime[i]]
                state.unassigned_order_staytime = state.unassigned_order_staytime + [state.order_stay_time[i]]
                state.unassigned_order_timewindow = state.unassigned_order_timewindow + [state.order_timewindow[i]]
                state.change_elapse = state.change_elapse + [True]
                g.node_all_orders[state.unassigned_order_node[i]] += 1
                g.add_order_node(state.order_node[i])

    if if_display_elapse: print('ordernodes',state.unassigned_order_node)
    state.previous_time = state.current_time
    if if_display_details:
        print('postman_prev_node', state.postman_prev_node)
        print('unassigned_order_node', state.unassigned_order_node)
        print('unassigned_order_staytime', state.unassigned_order_staytime)
    state.postman_elapse = [None for _ in range(PN)]
    if len(state.unassigned_order_popuptime) == 0:
        if if_display_details: print('exit2')
        state.current_time = state.current_time + state.elapse
        return g, state
    
    if_postman_unassigned = [x == None for x in
                             state.postman_destination_node]  # A postman is idling if his current destination is None
    
    if if_display_details: print(if_postman_unassigned)
    if_assignment_required = [False for _ in range(NN)]  # this is for destinations
    delete_order = []
    
    # with respect to eah individual postman, can triger the recursion
    unassigned_node_set = list(set(state.unassigned_order_node))
    if if_display_details: print('non repeating', unassigned_node_set)
    withhold_orders = []  # indices of orders to be withheld. orders shall not be assigned because postmen are there.
    withhold_locations = []
    # -------------------------------------- Close proximity assignment----------------------------
    #print('order node',state.unassigned_order_node)
    for unassigned_node in unassigned_node_set:
        all_indices = [index for index, element in enumerate(state.unassigned_order_node) if element == unassigned_node]
        if all_indices == []:  # check module
            print('error, empty indices list')
        # indices of all unassigned orders with this element
        all_popuptimes = [element for index, element in enumerate(state.unassigned_order_popuptime) if
                          index in all_indices]  # find the popuptime of all customers in this region
        all_timewindow = [element for index, element in enumerate(state.unassigned_order_timewindow) if
                          index in all_indices]

        earliest_order = all_indices[int(np.argmin(all_popuptimes))]
        latest_order=all_indices[int(np.argmax(all_popuptimes))]
            # find the earliest customer, only assign this one, others put aside

        
        if unassigned_node not in state.postman_destination_node:  # if not assigned, then we update this value
            state.node_earliest_popuptime[unassigned_node] = min(all_popuptimes)
            state.node_earliest_timewindow[unassigned_node] = all_timewindow[int(np.argmin(all_popuptimes))]
            state.node_latest_popuptime[unassigned_node]=max(all_popuptimes)
            state.node_latest_timewindow[unassigned_node]=all_timewindow[int(np.argmax(all_popuptimes))]
            
        if unassigned_node in state.postman_prev_node:  # state.postmaninitialnode?
            for p_idx in range(PN):
                if state.postman_destination_node[p_idx] == None and state.postman_prev_node[p_idx] == unassigned_node:
                    if if_display_details: print('passed criteria')
                    # 2 conditions
                    # The initial node of postman coincides with the destination node
                    # postman is not assigned any task
                    withhold_locations.append(state.unassigned_order_node[earliest_order])  # cannot be moved to the higher level,
                    # this is under the assumption that a postman is assigned to this location
                    distance = 0
                    state.postman_destination_node[p_idx] = int(state.unassigned_order_node[earliest_order])
                    state.postman_destination_indice[p_idx] = int(state.unassigned_order_indice[earliest_order])
                    if_postman_unassigned[p_idx] = False
                    delete_order.append(earliest_order)  # delete already assigned
                    state.postman_stayingtime[p_idx] = state.unassigned_order_staytime[earliest_order]
                    assignment_time = max(state.current_time, state.unassigned_order_popuptime[earliest_order])
                    state.elapse = min(state.elapse, state.unassigned_order_staytime[earliest_order])
                    if if_display_elapse: print('samenode elapse',state.elapse,state.unassigned_order_staytime[earliest_order])
                    g.add_assignment(p_idx, int(state.unassigned_order_node[earliest_order]),
                                     assignment_time, distance, state.unassigned_order_popuptime[earliest_order],
                                     state.unassigned_order_indice[earliest_order],
                                     state.unassigned_order_timewindow[earliest_order], 'close',
                                     state.unassigned_order_staytime[earliest_order])
                    postman_status[p_idx]=1
                    state.exp_travel_time[p_idx]=distance/speed
                    state.postman_current_path_distance[p_idx] = distance
                    if if_display_details:
                        print('-samenode-', 'assignpostman', p_idx, 'to destination',
                              int(state.unassigned_order_node[earliest_order]), 'starting from',
                              state.postman_prev_node[p_idx], 'at', assignment_time,
                              'popup', state.unassigned_order_popuptime[earliest_order],
                              'staytime', state.unassigned_order_staytime[earliest_order])
                    g.update_postman(None, None, p_idx, 1, 0)  # change status to working
                    state.postman_assignment_time[p_idx] = assignment_time
                    break
    state.unassigned_order_node = [int(item) for idx, item in enumerate(state.unassigned_order_node) if
                                   idx not in delete_order]
    state.unassigned_order_indice = [int(item) for idx, item in enumerate(state.unassigned_order_indice) if
                                     idx not in delete_order]
    state.unassigned_order_popuptime = [item for idx, item in enumerate(state.unassigned_order_popuptime) if
                                        idx not in delete_order]
    state.unassigned_order_timewindow = [int(item) for idx, item in enumerate(state.unassigned_order_timewindow) if
                                         idx not in delete_order]
    state.unassigned_order_staytime = [int(item) for idx, item in enumerate(state.unassigned_order_staytime) if
                                       idx not in delete_order]
    NN_unassigned_orders = []  # list of all unassigned orders sent to the neural network
    for i, node in enumerate(state.unassigned_order_node):
        if node in state.postman_destination_node or node in withhold_locations:  # part of the destinations
            withhold_orders.append(i)
        else:  # only those destinations with no postman heading towards shall be assigned a new postman. other orders should withhold.
            NN_unassigned_orders.append(node)  # unassigned order index
    # -------------------------------------end of proximity assignment----------------------------------
    # -----------------------------preprocess NN assignment and greedy assignment-----------------------
    num_vertices = g.count_vertices()
    x_edges, x_edges_values = g.get_adjacency()  # adjaency matrix, edge distances
    # x_edges in a form of 0 or 1, x_edges_values are in a form of distance
    # if_order = [0 for _ in range(num_vertices)]  # if this customer destination requires assignment7
    postman_np = np.zeros((1, PN, postman_dim-3))
   # postman_feature = torch.zeros(1, PN, postman_dim).to(device)
    # postman feature: current location coordinates, which node it is heading toward
    # it is currently in between which sets of coordinates. already traversed percentage
    for i in range(PN):  # assignment is also made during the initial stage
        postman_np[0, i, 0] = postmanx[i]/Mag_Factor  # x coordinate
        postman_np[0, i, 1] = postmany[i]/Mag_Factor  # y coordinate
        postman_np[0, i, 4] = postman_status[i]  # idling or working (either servering or on the way) 0 for idling
        postman_np[0, i, 5] = state.current_time / totaltime  # current time
        postman_np[0, i, 6] = state.exp_travel_time[i] / 20  # expected travel time
        if postman_status[i] == 1:  # it is working
            if state.postman_destination_node[i] != None:#it is walking
                destin=state.postman_destination_node[i]
                postman_np[0, i, 2] = xcoords[destin] / Mag_Factor # destination coordinates
                postman_np[0, i, 3] = ycoords[destin] / Mag_Factor # destination coordinates
                postman_np[0, i, 7] = (state.current_time - state.node_earliest_popuptime[destin] -
                                            state.node_earliest_timewindow[destin]) / 50
                if xcoords[destin]==postmanx[i] and ycoords[destin]==postmany[i]:
                    postman_np[0, i, 8] = 1  # has arrived and working there
        else:#it is idling
            postman_np[0, i, 2] = postmanx[i]/Mag_Factor
            postman_np[0, i, 3] = postmany[i]/Mag_Factor
            if state.postman_destination_node[i] != None:
                print(iteration)
                print(postman_status)
                print(state.postman_destination_node)
                print('wrong, destination does not equal to None')
                sys.exit()
    node_postman = {i:[] for i in range(NN)}
    x_nodes_np = np.zeros((1, NN, node_dim-6))
    x_nodes_np[0, :, 0] = [xcoords[k]/Mag_Factor for k in range(len(xcoords))]  # x coordinates
    x_nodes_np[0, :, 1] = [ycoords[k]/Mag_Factor for k in range(len(ycoords))]  # y coordinates
    df = pd.DataFrame(
        {'o_idx': list(range(len(state.unassigned_order_node))),  # o_id:  order index in the unassignedorder list
         'n_id': state.unassigned_order_node,  # node id of unassigned order
         'pop_t': state.unassigned_order_popuptime})  # popup time of unassigned order
    earliest_order_indices = {n_id: group.sort_values(by='pop_t')['o_idx'].tolist()[0] for n_id, group in
                              df.groupby(by='n_id')}
    for node in state.unassigned_order_node:  # find the latest appearance time
        if node not in state.postman_destination_node and node not in withhold_locations:
            if_assignment_required[int(node)] = True
        x_nodes_np[0, node, 2] += 1/state.PN  # add one unassigned order # if no customer leave it as zero.

    for i in range(NN):  # delivered orders will also be counted here.
        if i in g.current_order_node:#include all unfinished delivery nodes and unassigned deliver nodes
            x_nodes_np[0, i, 3] = (state.current_time - state.node_earliest_popuptime[i] -
                  state.node_earliest_timewindow[i]) / 100  # undelivered order, what time deadline would arrive. no need for recursion
            #can possibly be late, therefore can be negative
        x_nodes_np[0, i, 5] = g.node_all_orders[i]/(totaltime/4) #total number of appeared orders. No need for recursion
    for i in g.current_order_node:
        x_nodes_np[0, i, 6]+=1/state.PN #total number of undelivered nodes, including all unfinished and unassigned nodes. No need for recursion
    
    for i in range(PN):
        destin = state.postman_destination_node[i]
        if destin != None:
            x_nodes_np[0, destin, 4] += 1/state.PN  # if it is assigned then it becomes one. need for recursion
            node_postman[destin].append(i)  # 可能会覆盖，todo:之后可以改进
            x_nodes_np[0, destin, 7] = state.exp_travel_time[i] / 20 #when it is negative, reflects how long it has arrived
            if state.exp_travel_time[i]<=0:# it has arrived and still working on it
                x_nodes_np[0,destin,8]=1
                if postman_status[i]!=1:
                    print('error, postman is not working')
                    raise KeyboardInterrupt
    for i in range(NN):
        x_nodes_np[0, i, 9]=state.current_time / totaltime
    # todo: add the forcast of each node here
    if is_predict:
        predict = state.predict_order_volume(NN, is_predict)
        predict = torch.from_numpy(predict).to(device)
        predict = predict.unsqueeze(0)
    all_adjacency_np = np.expand_dims(2 * np.identity(NN + PN), axis=0)
    all_adjacency_np[0, 0:NN, 0:NN] = x_edges + np.identity(NN)
    allOD_np = np.zeros([1, PN + NN, PN + NN])
    allOD_np[0, 0:NN, 0:NN] = x_edges_values
    closest_node={}
    closest_distance={}
    second_closest_node={}
    second_closest_distance={}
    for i in range(PN):
        for j in range(NN):  # problem: cannot tell distance of zero and particular distances
            if state.all_edges != [] and state.all_edges[i] != None and j in state.all_edges[i] and \
                    state.postman_prev_node[i] != state.postman_destination_node[
                i]:  # if this node is one of the two edge nodes
                all_adjacency_np[0, NN + i, j] = 3 #3 for connecting postman and the customer locations
                all_adjacency_np[0, j, NN + i] = 3
                if j == state.all_edges[i][0]:
                    allOD_np[0, NN + i, j] = postman_percentage[i] * x_edges[state.all_edges[i][0], state.all_edges[i][1]]
                    allOD_np[0, j, NN + i] = postman_percentage[i] * x_edges[state.all_edges[i][0], state.all_edges[i][1]]
                    if postman_percentage[i]<=0.5:
                        closest_node[i]=state.all_edges[i][0]
                        second_closest_node[i]=state.all_edges[i][1]
                        closest_distance[i]=postman_percentage[i] * x_edges[state.all_edges[i][0], state.all_edges[i][1]]
                        second_closest_distance[i] = (1-postman_percentage[i]) * x_edges[state.all_edges[i][0], state.all_edges[i][1]]
                        assert second_closest_distance[i]>closest_distance[i], 'err, closest distance wrong 1'
                    else:
                        closest_node[i]=state.all_edges[i][1]
                        second_closest_node[i]=state.all_edges[i][0]
                        closest_distance[i] = (1-postman_percentage[i]) * x_edges[
                            state.all_edges[i][0], state.all_edges[i][1]]
                        second_closest_distance[i]=postman_percentage[i]*x_edges[state.all_edges[i][0], state.all_edges[i][1]]
                        assert second_closest_distance[i] > closest_distance[i], 'err, closest distance wrong 2'
                else:
                    allOD_np[0, NN + i, j] = (1 - postman_percentage[i]) * x_edges[
                        state.all_edges[i][0], state.all_edges[i][1]]
                    allOD_np[0, j, NN + i] = (1 - postman_percentage[i]) * x_edges[
                        state.all_edges[i][0], state.all_edges[i][1]]
                    if postman_percentage[i]>=0.5:
                        closest_node[i]=state.all_edges[i][0]
                        second_closest_node[i] = state.all_edges[i][1]
                        closest_distance[i] = (1-postman_percentage[i]) * x_edges[state.all_edges[i][0], state.all_edges[i][1]]
                        second_closest_distance[i]=postman_percentage[i] * x_edges[state.all_edges[i][0], state.all_edges[i][1]]
                        assert second_closest_distance[i] > closest_distance[i], 'err, closest distance wrong 3'
                    else:
                        closest_node[i]=state.all_edges[i][1]
                        second_closest_node[i] = state.all_edges[i][0]
                        closest_distance[i] = postman_percentage[i] * x_edges[state.all_edges[i][0], state.all_edges[i][1]]
                        second_closest_distance[i] = (1-postman_percentage[i]) * x_edges[state.all_edges[i][0], state.all_edges[i][1]]
                        assert second_closest_distance[i] > closest_distance[i], 'err, closest distance wrong 4'
            if j == state.postman_prev_node[i] and (state.postman_destination_node[i] == None or j == state.postman_destination_node[i]):
                # right at the location
                all_adjacency_np[0, NN + i, j] = 3
                all_adjacency_np[0, j, NN + i] = 3
                closest_node[i] = j
                second_closest_node[i] = j
                closest_distance[i] = 0
                second_closest_distance[i]=0
                allOD_np[0, NN + i, j] = 0
                allOD_np[0, j, NN + i] = 0
    for i in range(PN):
        for j in range(NN):
            if j!=closest_node[i] and j!=second_closest_node[i]:
                allOD_np[0,NN+i,j]=min(closest_distance[i]+g.findOD(closest_node[i],j),
                                    second_closest_distance[i]+g.findOD(second_closest_node[i],j))
                allOD_np[0,j,NN+i]=allOD_np[0,NN+i,j]
        for k in range(PN):
            if i==k:
                allOD_np[0, NN + i, NN + k]=0
            else:
                allOD_np[0,NN+i,NN+k]=min(closest_distance[i]+g.findOD(closest_node[i],closest_node[k])+closest_distance[k],
                                   second_closest_distance[i]+g.findOD(second_closest_node[i],closest_node[k])+closest_distance[k],
                                   closest_distance[i]+g.findOD(closest_node[i],second_closest_node[k])+second_closest_distance[k],
                                   second_closest_distance[i]+g.findOD(second_closest_node[i],second_closest_node[k])+second_closest_distance[k])
                allOD_np[0,NN+k,NN+i]=allOD_np[0,NN+i,NN+k]
            
    # -----------------------------------------Greedy assignment--------------------------------------------------
    locations_to_delete = []
    # three scenarios: 1. initial 2. only one unassigned postman, one assignment required. 3. more than one postmen,  multiple assignments
    # 已经有快递员的node，就不管（即不进行分配）；一对一，也就是一个快递，一个订单，直接匹配；多个快递员，一个订单，那么找最近的快递员分配给这个订单； 地图上只剩最后一个顾客，这种情况要单独处理；
    ## -----------------------------------------End of greedy assignment-------------------------------------------
    # -------------------------------------------Neural Network---------------------------------------------------
    # else: # 除了以上的情况，都用神经网络来做
    NNassignment = {}
    if_all_no_assignment=False
    if sum(if_postman_unassigned) > 0 and sum(if_assignment_required) > 0:
        if len(state.order_indice) == 0:  # about to end, assignment will be made at once.
            if sum(if_postman_unassigned)>=len(set(NN_unassigned_orders)):
                Reject_NA_Rate1 = 1
            else:
                Reject_NA_Rate1 = Reject_NA_Rate
        else:
            if max(state.order_indice) in state.unassigned_order_indice and len(NN_unassigned_orders)<=sum(if_postman_unassigned):
                Reject_NA_Rate1 = 1
            else:
                Reject_NA_Rate1 = Reject_NA_Rate
        if len(state.order_popuptime)!=0:
            if max(state.order_popuptime)<=state.current_time+0.01 and is_all_none(state.postman_destination_node):
                Reject_NA_Rate1=1
        else:
            if is_all_none(state.postman_destination_node):
                Reject_NA_Rate1=1
        if Greedyprocess:
            Reject_NA_Rate1=0
        state.RL_decisions += 1
        NNassignment1, NNlogprobs, Entropy, state.prevemb, additional_penalty = \
            neuralnetwork(all_adjacency_np, allOD_np, x_nodes_np, postman_np, if_assignment_required,
                          # 需要告诉model哪些是有用的，图神经网络会被每一个节点学习embedding，但是不是所有的节点嵌入都有用
                          if_postman_unassigned, state.prevemb,  # 这里的state prevemb现在没有用到
                          state, g, node_postman, Reject_NA_Rate1,Greedyprocess,
                          NNassignment=None, NNlogprobs=None, Entropy=None, recursion=False, istraining=istraining,batch=batch)
        g.add_additional_penalty(additional_penalty)
        if if_display_details:
            print(NNassignment1)
            print(NNassignment1[0].items())
        for i in list(NNassignment1[0]):
            NNassignment[i] = NNassignment1[0][i]
        
        if batch != None and is_training:
            buffer.logprobs[batch].append(torch.clamp(NNlogprobs[0], -100, 0).to(device))
            buffer.entropy[batch].append(Entropy[0].to(device))
          #  print('loc1',batch,buffer.logprobs[batch])
        for i in NNassignment1[0].values():
            if i != None:
                if_assignment_required[i] = False
        allpostmen = NNassignment.keys()
        # NNassignment:dictionary left hand side postman index, righthand side is the destination index
        # list of postman
        # location numbers not indices
        if_all_no_assignment=True
        skipped_order_popuptime=state.unassigned_order_popuptime.copy()
        
        for i in allpostmen:  # NNassignment[i] is the destination
            if NNassignment[i]==None and is_training!=True:
                Num_Nodispatch[batch]+=1
            if NNassignment[i] != None:
                state.postman_destination_node[i] = NNassignment[i]
                # evaluate the expected arrival time. Then if change elapse is needed, go backwards in time
                D, _ = g.find_distance(state.postman_prev_node[i], NNassignment[i])
                duration = D / speed
                if_all_no_assignment=False
                all_location_indices = [j for j in range(len(state.unassigned_order_node)) if
                                        state.unassigned_order_node[j] == NNassignment[i]]
                all_corresponding_popuptimes = [state.unassigned_order_popuptime[j] for j in all_location_indices]
                assigned_index = all_location_indices[np.argmin(all_corresponding_popuptimes)]
                locations_to_delete.append(assigned_index)
                assignment_time = max(state.current_time, state.node_earliest_popuptime[NNassignment[i]])
                skipped_order_popuptime.remove(state.node_earliest_popuptime[NNassignment[i]])
                if state.node_earliest_popuptime[NNassignment[i]] > state.current_time:  # check module
                    print('error, future order pop up')
                    sys.exit()
                g.add_assignment(i, int(NNassignment[i]), assignment_time, D,
                                 state.node_earliest_popuptime[NNassignment[i]],
                                 state.unassigned_order_indice[earliest_order_indices[int(NNassignment[i])]],
                                 state.unassigned_order_timewindow[earliest_order_indices[int(NNassignment[i])]], 'RL',
                                 state.unassigned_order_staytime[earliest_order_indices[int(NNassignment[i])]])
                state.postman_destination_indice[i] = state.unassigned_order_indice[
                    earliest_order_indices[int(NNassignment[i])]]
                state.postman_current_path_distance[i] = D
                state.postman_stayingtime[i] = state.unassigned_order_staytime[
                    earliest_order_indices[int(NNassignment[i])]]
                state.postman_assignment_time[i] = assignment_time
                state.elapse = min(state.elapse, duration + state.unassigned_order_staytime[
                    earliest_order_indices[int(NNassignment[i])]])
                if if_display_elapse: print('NN elapse',state.elapse, duration+state.unassigned_order_staytime[
                    earliest_order_indices[int(NNassignment[i])]])
                # -------------------------------assigned index is the unassigned postman index----------------------------
            else:
                pass
    # -------------------temporary exit command--------------------------
    # check if there are duplicates in a list
    a_set = set(locations_to_delete)
    contains_duplicates = len(locations_to_delete) != len(a_set)
    state.unassigned_order_node = [state.unassigned_order_node[i] for i in range(len(state.unassigned_order_node)) if
                                   i not in locations_to_delete]
    state.unassigned_order_indice = [state.unassigned_order_indice[i] for i in range(len(state.unassigned_order_indice))
                                     if i not in locations_to_delete]
    state.unassigned_order_popuptime = [state.unassigned_order_popuptime[i] for i in
                                        range(len(state.unassigned_order_popuptime)) if i not in locations_to_delete]
    state.unassigned_order_timewindow = [state.unassigned_order_timewindow[i] for i in
                                         range(len(state.unassigned_order_timewindow)) if i not in locations_to_delete]
    state.unassigned_order_staytime = [state.unassigned_order_staytime[i] for i in
                                       range(len(state.unassigned_order_staytime)) if i not in locations_to_delete]
    if if_all_no_assignment:
        max_skipped_time=max(skipped_order_popuptime)
        remaining_order_popuptimes=[state.order_popuptime[i] for i in range(len(state.order_popuptime)) if state.order_popuptime[i]>max_skipped_time
                                    and state.order_popuptime[i]>state.current_time]
        if len(remaining_order_popuptimes)!=0:
            planned_elapse=min(remaining_order_popuptimes)-state.current_time
            if planned_elapse!=0:
                state.elapse=min(state.elapse,planned_elapse)
        if max_skipped_time==max(state.unassigned_order_popuptime) and len(state.order_popuptime)==0 and is_all_none(state.postman_destination_node):
            state.elapse=0
            print('trigger1')
            print(state.unassigned_order_popuptime)
            print(NN_unassigned_orders)
            print(sum(if_postman_unassigned))
            raise KeyboardInterrupt

    state.current_time = state.current_time + state.elapse
    assert len(state.postman_destination_node) == len(state.postman_prev_node), "Error,wrong destination size"
    if if_display_elapse: print('finalelapse',state.elapse)
    return g, state

def calculate_and_log_reward(g: Env.Graph, log_config: dict,PN=0):
    """
    calcuate the reward, and then save the details into the desk
    :param g:
    :param log_config: {"fout": str, "sheet_name": str, "server_version":bool, "time":float, 'rl_decision_num': int}
    if log_config['fout'] == 'not_save', then we don't save the details
    :return:
    """
    postman_assigned_nodes, postman_assigned_time, postman_assigned_distance, postman_assigned_popuptime, \
    postman_assigned_index, postman_assigned_timewindow, assignment_method, postman_assigned_staytime ,assignment_time_order \
        = g.get_assigned()
    # self.postman_delivered_nodes, self.postman_delivered_times, self.postman_delivered_index
    
    postman_delivered_nodes, postman_delivered_times, postman_delivered_index = g.get_delivered()
    
    for i in range(PN):
        if len(postman_assigned_index[i])!=0:
            postman_assigned_nodes[i] = sort_together([assignment_time_order[i], postman_assigned_nodes[i]])[1]
            postman_assigned_time[i] = sort_together([assignment_time_order[i], postman_assigned_time[i]])[1]
            postman_assigned_popuptime[i] = sort_together([assignment_time_order[i], postman_assigned_popuptime[i]])[1]
            postman_assigned_distance[i] = sort_together([assignment_time_order[i], postman_assigned_distance[i]])[1]
            assignment_method[i] = sort_together([assignment_time_order[i], assignment_method[i]])[1]
            postman_delivered_times[i] = sort_together([assignment_time_order[i], postman_delivered_times[i]])[1]
            postman_assigned_timewindow[i] = sort_together([assignment_time_order[i], postman_assigned_timewindow[i]])[1]
            postman_assigned_staytime[i] = sort_together([assignment_time_order[i], postman_assigned_staytime[i]])[1]
            postman_delivered_index[i] = sort_together([assignment_time_order[i], postman_delivered_index[i]])[1]
            postman_assigned_index[i] = sort_together([assignment_time_order[i], postman_assigned_index[i]])[1]
    Avg_Idle_Time1=0
    for i in range(PN):
        if len(postman_assigned_time[i])!=0:
            Avg_Idle_Time1+=postman_assigned_time[i][0] + \
            sum([postman_assigned_time[i][j + 1] - postman_delivered_times[i][j] for j in
                 range(len(postman_assigned_time[i]) - 1)])

    episode_reward, distance_penalty, averagelateness, variance = Env.episode_reward(popup_times=postman_assigned_popuptime,
                                        postman_assigned_index=postman_assigned_index,
                                        delivered_times=postman_delivered_times,
                                        postman_delivered_index=postman_delivered_index,
                                        time_windows=postman_assigned_timewindow,
                                        staying_times=postman_assigned_staytime,
                                        assigned_distance=postman_assigned_distance)
    episode_reward=episode_reward+g.add_penalty
    if log_config['fout'] == 'not_save': return episode_reward, distance_penalty # directly return the reward without saving the details
    action_length = len(flatten(postman_assigned_nodes))
    evaluationtime = [log_config['time']] + [0 for _ in range(action_length - 1)]
    episodereward1 = [episode_reward] + [0 for _ in range(action_length - 1)]
    rl_decision_num = log_cfg.get('rl_decision_num', 0)
    RL_decision_num = [rl_decision_num] + [0 for _ in range(action_length - 1)]
    dfdict = {'assigned tasks': flatten(postman_assigned_nodes),
              'assigned time': flatten(postman_assigned_time),
              'request time': flatten(postman_assigned_popuptime),
              'timewindows': flatten(postman_assigned_timewindow),
              'stayingtime': flatten(postman_assigned_staytime),
              'assigned distance': flatten(postman_assigned_distance),
              'assignedindex': flatten(postman_assigned_index),
              'assignmentmethod': flatten(assignment_method),
              'delivered times': flatten(postman_delivered_times),
              'deliveredindex': flatten(postman_delivered_index),
              'episodereward': episodereward1,
              'evaluationtime': evaluationtime,
              'RL decisions': RL_decision_num}
    df = pd.DataFrame(data=dfdict)
    if len(postman_assigned_popuptime) != len(postman_delivered_times) or len(postman_delivered_nodes) != len(postman_delivered_times):
        print('error, wrong size assigned and delivered tasks')
        print('postmanassignedtasks', len(postman_assigned_nodes), 'postmandeliveredtasks', len(postman_delivered_nodes))
        sys.exit()
    Avg_Travel_Dist1=sum([sum(postman_assigned_distance[i]) for i in range(PN)])
    return episode_reward, distance_penalty, averagelateness, variance, rl_decision_num, Avg_Idle_Time1, Avg_Travel_Dist1

global distancepenaltyratio
if is_training:
    distancepenaltyratio=args['Distancepenalty']
    distancepenaltydecay = 1e-2
else:
    distancepenaltyratio=0
    distancepenaltydecay=0


def rl_batch_kernal(args: dict):
    ep, batch, lb, hb, NN, PN, batch_size, training_size =  args['ep'], args['batch'], args['lb'], args['hb'], args['NN'], args['PN'], args['batch_size'], args['training_size']
    device = args['device']
    g4, state4 = args['g'], args['state']#, args['net']
    print(f'1Episode:{ep} | Batch:{batch}')
    np.random.seed()
    rd = np.random.uniform()
    if rd <= 0.1:
        Greedyprocess = True
        print('Greedyprocess')
    else:
        Greedyprocess = False
    start_time = time.time()
    g4, state4 = RLpolicy(g4, PN, state4, (ep % training_size) * batch_size + batch, 0,Reject_NA_Rate,
                          initial=True, neuralnetwork=actor_net, NN = NN, device = device,Greedyprocess=Greedyprocess)
    g4, state4 = Env.Environment(g4, state4, initial=True)
    g4, state4 = RLpolicy(g4, PN, state4, (ep % training_size) * batch_size + batch, 1,Reject_NA_Rate,
                          initial=False, neuralnetwork=actor_net, NN = NN, device = device,Greedyprocess=Greedyprocess)
    RLcounter = 0
    state4.Done = False
    # -------------------------end of RL initialisation----------------------------
    # --------------------------start of RL policy----------------------------------
    while state4.Done != True:
        g4, state4 = Env.Environment(g4, state4, initial=False)
        if state4.Done: break
        state4.update_postman_xy(g4)
        g4, state4 = RLpolicy(g4, PN, state4, (ep % training_size) * batch_size + batch, RLcounter + 2,Reject_NA_Rate,
                              initial=False, neuralnetwork=actor_net, NN = NN, device = device,Greedyprocess=Greedyprocess)
        RLcounter += 1
    log_cfg = {'fout': 'not_save'} #多个线程操作同一个文件可能会有问题，所以这里不写文件了
    episode_reward, distance_penalty,_,_,_,_,_ = calculate_and_log_reward(g4, log_cfg, PN)
    buffer.add_new_instance(state4.RL_decisions, episode_reward+distancepenaltyratio*distance_penalty) # todo:全局变量添加
    result = {'episode_reward': episode_reward, 'RL_decisions': state4.RL_decisions}
    return result


def rl_batch_kernal_2(args: dict):
    ep, batch, lb, hb, NN, PN, batch_size, training_size = args['ep'], args['batch'], args['lb'], args['hb'], args['NN'], args['PN'], args['batch_size'], args['training_size']
    device = args['device']
    trainingbatch = args['trainingbatch']
    print('trainingbatch', trainingbatch)
    g1, state1 = args['g'], args['state']
    np.random.seed()
    rd = np.random.uniform()
    Greedyprocess = rd <= args['Greedyprocessratio']
    start_time = time.time()
    g1, state1 = RLpolicy(g1, PN, state1, trainingbatch, 0, Reject_NA_Rate,
                          initial=True, neuralnetwork=actor_net, NN=NN, device=device, Greedyprocess=Greedyprocess)
    g1, state1 = Env.Environment(g1, state1, initial=True)
    g1, state1 = RLpolicy(g1, PN, state1, trainingbatch, 1, Reject_NA_Rate,
                          initial=False, neuralnetwork=actor_net, NN=NN, device=device, Greedyprocess=Greedyprocess)
    RLcounter = 0
    state1.Done = False
    # -------------------------end of RL initialisation-----------------------------
    # --------------------------start of RL policy----------------------------------
    
    while state1.Done != True:
        g1, state1 = Env.Environment(g1, state1, initial=False)
        if state1.Done: break
        state1.update_postman_xy(g1)
        g1, state1 = RLpolicy(g1, PN, state1, trainingbatch, RLcounter + 2, Reject_NA_Rate,
                initial=False, neuralnetwork=actor_net, NN=NN, device=device, Greedyprocess=Greedyprocess)
        RLcounter += 1
    log_cfg = {'fout': path + 'RLStatistics.xlsx', 'sheet_name': f'graph_{graph_seed}',
               'server_version': serverversion, 'time': time.time() - start_time, 'rl_decision_num': state1.RL_decisions}
    episode_reward, distance_penalty,_,_,_,_,_ = calculate_and_log_reward(g1, log_cfg, PN)
    if not is_training:
        RL_test_rewards.append(episode_reward)
    buffer.add_new_instance(state1.RL_decisions, episode_reward + distancepenaltyratio * distance_penalty)
    result = {'episode_reward': episode_reward, 'RL_decisions': state1.RL_decisions}
    state1.reset()
    g1.reset(NN)  # reset the graph
    return result

if __name__ == '__main__':
    pprint(args)
    # -----------------------------------set device to CPU or GPU-----------------------------------
    # Define optimizer
    optimizer = optim.Adam(actor_net.parameters(), lr=learningrate)
    torch.autograd.set_detect_anomaly(True)
    try:
        actor_net.load_state_dict(torch.load(actordict))
        print('loaded parameters')
    except:
        if is_training == False: # when evaluate, the model must be loaded.
            print('no actor dict loaded')
            sys.exit()
        else:
            print('initialising random parameters')
            
    actor_net_target=copy.deepcopy(actor_net)
    graph_seed = 0
    num_graphs = 1# num graphs：how many graphs to be generated
    num_scenarios = 1 if PPO else 5 # how many times a scenario is trained

    if is_training == False:
        num_scenarios = 1
        batch_size = 1
    trainingsteps = 0
    entropyrate = args['entropyrate']
    # global greedythreshold
    percentage_improvements=[]
    all_losses=[]
    RL_test_rewards=[]
    global Num_Nodispatch,Avg_Travel_Dist,Avg_Idle_Time
    Num_Nodispatch=[0 for i in range(num_episodes*num_graphs)]
    Avg_Travel_Dist=[]
    Avg_Idle_Time=[]
    Greedy_test_rewards=[]
    allentropies=[]
    allRLdecisionnums=[]
    allstaytime=[]
    
    if is_training: allepisodes=flatten([[2*i,2*i+1,2*i,2*i+1] for i in range(num_episodes//2)])
    else:
        allepisodes = [i+1000 for i in range(num_episodes)]
        alllateness=[]
        allvar=[]
        allGreedylateness = []
        allGreedyvar = []
        Greedytime=[]
        RLtime=[]
    try:
        greedy_optimiser_rewards = []
        greedy_rollout_rewards = []
       # print(num_episodes*num_graphs)
        #sys.exit()
        while graph_seed < num_graphs: #num graphs：要生成多少个graph，gr/aphseed是生成graph的随机种子
            trainingbatch = 0
            print('newgraph with seed',graph_seed)
            for ep in allepisodes:
                if ep==num_episodes and graph_seed==num_graphs:
                    raise KeyboardInterrupt
                if ep % training_size==0: # including the first one
                    greedy_rollout_rewards=[]
                    if is_training: greedy_optimiser_rewards=[]
                #for scenario in range(num_scenarios):
                RL_episode_reward = 0
                # -----------------------generate random problems-------------------------
                if ep % 2 == 0 and is_training:
                    print('saving parameters')
                    torch.save(actor_net.state_dict(), actordict)
                PN = PNlower if PNlower == PNhigher else np.random.randint(low=PNlower, high=PNhigher)
                NN = NNlower if NNlower == NNhigher else np.random.randint(low=NNlower, high=NNhigher) # todo: should change to poisson distribution
                g1 = Env.Graph(PN)  # initialise a graph
                g1 = Env.graph_generator(g1, NN, Mag_Factor, randomseed=graph_seed, graphname=None,Euclidean=Euclidean) #NN: number of nodes in the graph
                state1 = Env.State(PN, NN, device)  # redeclare state
                g1, state1 = Env.random_generator(g1, lb, hb, lam, NN, PN, state1, randomseed=ep)
                allstaytime.append(sum(state1.order_stay_time))
                state1.current_time = 0
                if display_myopic:
                    g2 = copy.deepcopy(g1)
                    state2 = copy.deepcopy(state1)
                    # --------------------end of random graph generation---------------------------
                    # ----------------------greedy initialisation----------------------------------
                    start_time = time.time()
                    print('----------run greedy policy-----------')
                    g2, state2 = Env.greedy_policy(state2, g2, PN, initial=True)
                    g2, state2 = Env.Environment(g2, state2, initial=True)
                    g2, state2 = Env.greedy_policy(state2, g2, PN, initial=False)
                    greedy_counter = 0
                    state2.Done = False
                    # ------------------------end of greedy initialisation----------------------------
                    # ------------------------run greedy policy---------------------------------------
                    # greedy policy does not require iterate over many parallel batches
                    while not state2.Done:
                        g2, state2 = Env.Environment(g2, state2, initial=False)
                        if state2.Done: break#@? 怎么样才算是状态结束？
                        state2.update_postman_xy(g2)#更新快递员位置坐标
                        g2, state2 = Env.greedy_policy(state2, g2, PN, initial=False)
                        
                        greedy_counter += 1
                    log_cfg = {'fout':path + 'GreedyStatistics.xlsx', "sheet_name":f'graph_{graph_seed}', 'server_version':serverversion, 'time': time.time() - start_time}
                    episode_reward,_,Greedyaveragelateness, Greedyvariance,_,_,_ = calculate_and_log_reward(g2, log_cfg, PN)
                    
                    greedy_optimiser_rewards.append(episode_reward)
                    if not is_training:
                        Greedy_test_rewards.append(episode_reward)
                        allGreedylateness.append(Greedyaveragelateness)
                        allGreedyvar.append(Greedyvariance)
                    print('Greedy reward', episode_reward)
                    state2.reset()
                    g2.reset(NN)  # reset the graph
                    g2, state2 = Env.random_generator(g2, lb, hb, lam, NN, PN, state2, randomseed=ep)
                    if not is_training: Greedytime.append(time.time()-start_time)
                    # --------------------------end of greedy policy---------------------------------
                #-----------------------------start of greedy RL rollout initialisation-----------------------------
                buffer.add_new_state(state1, g1)
                if is_training:
                    print('run greedy rollout rl')
                    g3 = copy.deepcopy(g1)
                    state3 = copy.deepcopy(state1)
                    g3, state3 = RLpolicy(g3, PN, state3, None, 0, Reject_NA_Rate,
                                          initial=True, neuralnetwork=actor_net_target, istraining=False, NN=NN, device=device)
                    g3, state3 = Env.Environment(g3, state3, initial=True)
                    g3, state3 = RLpolicy(g3, PN, state3, None, 1, Reject_NA_Rate,
                                          initial=False, neuralnetwork=actor_net_target, istraining=False, NN=NN, device=device)
                    greedy_counter = 0
                    state3.Done = False
                    while not state3.Done:
                        g3, state3 = Env.Environment(g3, state3, initial=False)
                        if state3.Done: break
                        state3.update_postman_xy(g3)
                        g3, state3 = RLpolicy(g3, PN, state3, None, greedy_counter + 2, Reject_NA_Rate,
                        initial=False, neuralnetwork=actor_net_target, istraining=False, NN = NN, device = device)
                        # 因为batch参数为None，所以不会对buffer进行操作
                        greedy_counter += 1
                    RLgreedyepisodereward,RLgreedy_distance_penalty= calculate_and_log_reward(g3, {'fout':"not_save"}, PN)
                    print('epsiodereward',RLgreedyepisodereward,'penalty',RLgreedy_distance_penalty)
                    
                #-----------------------------end of greedy RL rollout---------------------------------------
                # -------------------------RL initialisation-------------------------------------
                print('run rl')
                if display_VRPTW:
                    df, episode_reward = VRPTW.RecursiveReoptimisation(PN, NN, graph_seed, ep)
                    write_xlsx(df=df, fout=path + 'VRPTWStatistics.xlsx', sheet_name=f'graph_{graph_seed}', server_version=serverversion)
                if display_RL:
                    if not multithread:
                        for batch in range(batch_size):
                            if not is_training: Greedyprocess=False
                            else:
                                rd = np.random.uniform()
                                Greedyprocess = (rd <= args['Greedyprocessratio'])
                            if Greedyprocess:
                                print('greedyprocess')
                            print('Episode:', ep, 'batch', batch)
                            if is_training:
                                greedy_rollout_rewards.append(RLgreedyepisodereward+distancepenaltyratio*RLgreedy_distance_penalty)
                            start_time = time.time()
                            g1, state1 = RLpolicy(g1, PN, state1, trainingbatch, 0, Reject_NA_Rate,
                                                  initial=True, neuralnetwork=actor_net,istraining=is_training,
                                                  NN = NN, device = device, Greedyprocess=Greedyprocess)
                            g1, state1 = Env.Environment(g1, state1, initial=True)
                            g1, state1 = RLpolicy(g1, PN, state1, trainingbatch, 1, Reject_NA_Rate,
                                                  initial=False, neuralnetwork=actor_net,istraining=is_training,
                                                  NN = NN, device = device,Greedyprocess=Greedyprocess)
                            RLcounter = 0
                            state1.Done = False
                            # -------------------------end of RL initialisation-----------------------------
                            # --------------------------start of RL policy----------------------------------
                            while state1.Done != True:
                                g1, state1 = Env.Environment(g1, state1, initial=False)
                                if state1.Done: break
                                state1.update_postman_xy(g1)
                                g1, state1 = RLpolicy(g1, PN, state1, trainingbatch, RLcounter + 2,Reject_NA_Rate,
                                                      initial=False, neuralnetwork=actor_net,istraining=is_training,
                                                      NN = NN, device = device,Greedyprocess=Greedyprocess)
                                RLcounter += 1
                            trainingbatch += 1
                            log_cfg = {'fout':path+'RLStatistics.xlsx', 'sheet_name':f'graph_{graph_seed}',
                                       'server_version':serverversion, 'time': time.time() - start_time, 'rl_decision_num':state1.RL_decisions
                                       }
                            episode_reward, distance_penalty,averagelateness, variance,RL_decision_num,Avg_Idle_Time1,Avg_Travel_Dist1\
                                = calculate_and_log_reward(g1, log_cfg, PN)
                            if not is_training:
                                RL_test_rewards.append(episode_reward)
                                alllateness.append(averagelateness)
                                allvar.append(variance)
                                allRLdecisionnums.append(RL_decision_num)
                                Avg_Idle_Time.append(Avg_Idle_Time1)
                                Avg_Travel_Dist.append(Avg_Travel_Dist1)
                            buffer.add_new_instance(state1.RL_decisions, episode_reward+distancepenaltyratio*distance_penalty)
                            RL_episode_reward += episode_reward
                            state1.reset()
                            g1.reset(NN)  # reset the graph
                            g1, state1 = Env.random_generator(g1, lb, hb, lam, NN, PN, state1, randomseed=ep)
                            if not is_training: RLtime.append(time.time()-start_time)
                    #-------------------------------<
                    if multithread:
                        state1.reset()
                        g1.reset(NN)  # reset the graph
                        g1, state1 = Env.random_generator(g1, lb, hb, lam, NN, PN, state1, randomseed=ep)
                        args_lst = [{'ep':ep, 'batch':batch, 'g': copy.deepcopy(g1), 'state': copy.deepcopy(state1), #'net': copy.deepcopy(actor_net),
                            'lb':lb, 'hb':hb, 'NN': NN, 'PN':PN, 'batch_size':batch_size, 'training_size': training_size, 'device': device,
                            'trainingbatch': batch, 'Greedyprocessratio':args['Greedyprocessratio'],} for batch in range(batch_size)]
                        from threading import Thread
                        tasks = [Thread(target=rl_batch_kernal_2, args=(args,)) for args in args_lst]
                        for t in tasks:
                            t.start()
                        for t in tasks:
                            t.join()
                        if is_training:
                            for i in range(len(args_lst)):
                                greedy_rollout_rewards.append(RLgreedyepisodereward+distancepenaltyratio*RLgreedy_distance_penalty)
                    #--------------------->
                # ------------------------------end of RL policy--------------------------------
                # ----------------------------batch counter finishes----------------------------
                # ------------------------------update policy-----------------------------------
                if not is_training:
                    _, _, old_total_rewards, _, _ = buffer.get_instance()
                    print('oldtotalreward',old_total_rewards)
                    print(greedy_optimiser_rewards)
                    if len(greedy_optimiser_rewards)!=len(old_total_rewards):
                        print('error, wrong sizes')
                        raise KeyboardInterrupt
                    print('percentageimprovement',(-sum(old_total_rewards)+sum(greedy_optimiser_rewards))/(sum(greedy_optimiser_rewards)+0.1))
                if is_training and (ep+1) % training_size==0:
                    print('Enter Training')
                    print('episode', ep)
                    copy_actor = copy.deepcopy(actor_net)
                    _, _, old_total_rewards, _, old_logprobs = buffer.get_instance()
                    logprob_len=[len(old_logprobs[i]) for i in range(len(old_logprobs))]
                    selected = [i for i in range(len(logprob_len)) if logprob_len[i] > 0]
                    greedy_optimiser_rewards=[greedy_optimiser_rewards[i] for i in range(len(greedy_optimiser_rewards)) for j in range(batch_size+1)]
                    greedy_optimiser_rewards = [greedy_optimiser_rewards[i] for i in selected]
                    reward_mean = torch.FloatTensor(old_total_rewards).view(training_size, batch_size).mean(dim=1).unsqueeze(dim=1).repeat(1,batch_size).view(
                    training_size * batch_size).to(device)
                    reward_tensor = torch.FloatTensor(old_total_rewards)[selected].to(device)
                    reward_mean=reward_mean[selected]
                    greedyrolloutreward = torch.tensor(greedy_rollout_rewards)[selected].to(device)
                    oldlogprobs=torch.stack([torch.stack(old_logprobs[i]).sum() for i in range(len(old_logprobs)) if i in selected])
                    #print(torch.stack([torch.stack(buffer.entropy[i]).sum() for i in selected]).sum())
                    print(reward_tensor)
                    #print(reward_tensor.view(training_size,batch_size).mean(dim=1).unsqueeze(dim=1))
                    if GreedyRolloutBaseline:
                        print(greedyrolloutreward)
                        print(reward_tensor)
                        selected_logprobs =  (reward_tensor - greedyrolloutreward) * oldlogprobs
                    else:
                        selected_logprobs = (reward_tensor - reward_mean) * oldlogprobs
                    print('entropy',entropyrate*torch.stack([torch.stack(buffer.entropy[i]).sum() for i in selected]).sum())
                    percentage_improvement = (sum(greedy_optimiser_rewards) - sum(greedyrolloutreward.tolist())) / (sum(greedy_optimiser_rewards)+0.1)
                    percentage_improvements.append(percentage_improvement)
                    loss = selected_logprobs.mean() - entropyrate*torch.stack([torch.stack(buffer.entropy[i]).sum() for i in selected]).sum()
                    # direction should be along the positive size. The loss is supposed to be large negative
                    print('percentage improvement', percentage_improvement)
                    print('loss',loss)
                    optimizer.zero_grad()
                    # Calculate gradients
                    all_losses.append(loss.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_net.parameters(), 2, norm_type=2)
                    # Apply gradients
                    entropyrate-=entropystepdecay
                    entropyrate=max(0.02,entropyrate)
                    optimizer.step()
                    print('network parameters:')
                    for para in actor_net.parameters():
                        print('parameters:', para)
                        break
                    print('-' * 60)
                    # ------------------------------update counter----------------------------------
                    if softupdate:
                        #copy actor is the old one, actor_net is the latest one.
                        for actor_net1, copy_actor1 in zip(actor_net.parameters(), copy_actor.parameters()):
                            actor_net1.data.copy_((1-tau) * copy_actor1.data + tau * actor_net1.data)
                    if GreedyRolloutBaseline:
                        if trainingsteps % baseline_update_steps == 0 and trainingsteps != 0:
                            actor_net_target = copy.deepcopy(actor_net)
                    else:
                        actor_net_target = copy.deepcopy(actor_net)
                    print('update actor net target')
                    trainingsteps+=1
                    allentropies.append(torch.stack([torch.stack(buffer.entropy[i]).sum() for i in selected]).tolist()[0])
                    if trainingsteps % 10 == 0:
                        loss_records = pd.DataFrame(
                            {'all_losses': all_losses, 'percentage improvement': percentage_improvements,
                             'allentropies':allentropies})
                        loss_records.to_csv(path + 'loss.csv')
                    buffer.clear_logprobs()
                    buffer.clear_rewards()
                    distancepenaltyratio-=distancepenaltydecay
                    distancepenaltyratio=max(distancepenaltyratio,0)
                    Reject_NA_Rate-=Reject_decay
                    Reject_NA_Rate=max(Reject_NA_Rate,0.01)
                    
                    for g in optimizer.param_groups:
                        g['lr'] = (learningrate-5e-4)*(math.exp(-0.001*trainingsteps))+5e-4 #5e-4 at minimal value. decays with around 2000 training epoches
                    trainingbatch = 0
            graph_seed += 1

    except KeyboardInterrupt:
        print('saving parameters from keyboard interruption')

    torch.save(actor_net.state_dict(), actordict)
    if is_training:
        loss_records=pd.DataFrame({'all_losses': all_losses, 'percentage improvement': percentage_improvements,
                             'allentropies':allentropies})
        loss_records.to_csv(path+'loss.csv')
    else:
        if len(Greedy_test_rewards)>len(RL_test_rewards):
            del Greedy_test_rewards[-1]
        elif len(Greedy_test_rewards)<len(RL_test_rewards):
            del RL_test_rewards[-1]
        print(len(Greedy_test_rewards))
        print(len(RL_test_rewards))
        print(len(allRLdecisionnums))
        print(allRLdecisionnums)
        test_results=pd.DataFrame({'greedy':Greedy_test_rewards,'RL':RL_test_rewards,'lateness':alllateness,
             'var':allvar,'Greedylateness':allGreedylateness,'Greedyvar':allGreedyvar,'RLtime':RLtime,
             'Greedytime':Greedytime,'RLdecisionnum':allRLdecisionnums,'NumNodispatch':Num_Nodispatch,
             'allstaytime':allstaytime,'Avg_Traveldist':Avg_Travel_Dist,'Avg_Idle_Time':Avg_Idle_Time})
        test_results.to_csv(path+'test_results.csv')
    print('properly saved')
