# -*- coding: utf-8 -*-
import sys, torch, copy, time, os, time
import torch.nn as nn
from torch import optim
from openpyxl import load_workbook
import numpy as np
import pandas as pd
import copy

#--- own packages
import Parameters
import Environment as Env
import NetModel
from GreedyOptimisation import Greedyoptimiser
from more_itertools import sort_together
from utils import dir_check, flatten, write_xlsx
from pprint import pprint

global actor_net, is_training
#assignmentcounter = 0
args = Parameters.parameters()
args = vars(args)
pprint(args)
# ---------------control panel------------------------
is_training = True
display_myopic = True
PPO = False
display_RL = True
display_VRPTW=False
is_predict=False
Euclidean=0

if_display_details = False
serverversion=False
graph_seed = 0
customerseed = 5

local_time = time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime())
path = f'./output/{local_time}/'
dir_check(path)
actordict = path + 'actorreal1.pth'#神经网络参数
if display_VRPTW:
    import VRPTW
# -----------------hyperparameters--------------------
if display_myopic == False and display_RL == False:
    print('wrong, both displays are disabled')
    sys.exit()

global totaltime, speed, buffersize
NNlower = args['NNlower']
NNhigher = args['NNhigher']
speed = args['speed']
PNlower = args['PNlower']
PNhigher = args['PNhigher']
learningrate = args['learningrate']
lb = args['low_order_bound']
hb = args['high_order_bound']
totaltime = args['totaltime']
eps_clip = args['eps_clip']
training_size=args['training_size']
postman_dim=args['postman_dim']
node_dim=args['node_dim']
baseline_update_steps=args['baseline_update_steps']
if is_training == True:
    num_episodes = args['trainingepisode']
    batch_size = args['training_batch_size']
else:
    num_episodes = args['solvingepisode']
    batch_size = args['solving_batch_size']
buffersize = batch_size * training_size
samplesize = batch_size
gamma = args['gamma']
# -----------------------------------set device to CPU or GPU-----------------------------------
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")



def RLpolicy(g, PN, state, batch, iteration, initial=False,
             neuralnetwork=None,istraining=is_training):  # training indicates if training or playing experience
    if if_display_details:
        print('current time', state.current_time)
        print('len of order_popup_time', len(state.order_popuptime), 'len of order_node', len(state.order_node), 'len of unassigned_order_node',
              len(state.unassigned_order_node), 'len of unassigned_order_popuptime', len(state.unassigned_order_popuptime))
        print('postman_destination_node:', state.postman_destination_node)
    xcoords, ycoords = g.get_vertices()
    postmanx, postmany, postman_status, postman_percentage = g.get_postman()
   # print('initial',initial)
   # print('--------currenttime------',state.current_time)
    if initial:
        # init state information
        for i in range(len(state.order_popuptime)):
            if state.order_popuptime[i] <= 0:#主要是看等于0
                state.unassigned_order_node.append(state.order_node[i])
                state.unassigned_order_indice.append(state.order_indice[i])
                state.unassigned_order_popuptime.append(state.order_popuptime[i])
                state.unassigned_order_staytime.append(state.order_stay_time[i])
                g.add_order_node(state.order_node[i])
                state.node_earliest_popuptime[int(state.order_node[i])]=state.order_popuptime[i]
                state.unassigned_order_timewindow.append(state.order_timewindow[i])
             #   assignmentcounter.counter += 1
        #print('received',state.order_node)
        # init graph information
        for i in range(PN):
            g.update_postman(xcoords[int(state.postman_prev_node[i])], ycoords[int(state.postman_prev_node[i])], i, 0, 0)

        # for i in range(PN):
        #     if state.order_popuptime[i] <= 0:#主要是看等于0
        #         state.unassigned_order_node.append(state.order_node[i])
        #         state.unassigned_order_indice.append(state.order_indice[i])
        #         state.unassigned_order_popuptime.append(state.order_popuptime[i])
        #         state.unassigned_staying_time.append(state.order_stay_time[i])
        #         state.node_earliest_popuptime[int(state.order_node[i])]=state.order_popuptime[i]
        #         state.unassigned_order_timewindow.append(state.order_timewindow[i])
        #         assignmentcounter.counter += 1
        #     g.update_postman(xcoords[int(state.postman_prev_node[i])], ycoords[int(state.postman_prev_node[i])], i, 0, 0)
        if state.postman_destination_node == False or state.postman_destination_node == [] or state.postman_destination_node == None:
            state.postman_destination_node = [None for _ in range(PN)]
            state.postman_destination_indice = [None for _ in range(PN)]
        state.change_elapse = [False for _ in range(len(state.unassigned_order_node))]
        state.elapse=0 
    else:
        for i in range(len(state.order_popuptime)):
            if len(state.order_popuptime) == 0: break
            try:
                if state.order_popuptime[i] - state.current_time > 0:
                    state.elapse = min(state.elapse, state.order_popuptime[i] - state.current_time)
                    break
            except IndexError:
                state.elapse = min(state.elapse,totaltime)
        if state.elapse == totaltime:
            if len(state.unassigned_order_node) == 0 and len(state.order_node) == 0:  # all assignments are made
                state.Done = True
                if if_display_details: print('exit1')
                state.current_time = state.current_time + state.elapse
                return g, state
            elif len(state.unassigned_order_node) != 0:  # there are still remaining tasks at the same location the postman stays
                state.elapse = min(state.unassigned_order_staytime)
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
                g.add_order_node(state.order_node[i])
                state.change_elapse = state.change_elapse + [True]
             #   assignmentcounter.counter += 1
      
    state.previous_time=state.current_time
    if if_display_details:
        print('postman_prev_node', state.postman_prev_node)
        print('unassigned_order_node', state.unassigned_order_node)
        print('unassigned_staying_time', state.unassigned_order_staytime)
    state.postman_elapse = [None for _ in range(PN)]
    if len(state.unassigned_order_popuptime) == 0:
        if if_display_details: print('exit2')
        state.current_time = state.current_time + state.elapse
        return g, state

    if_postman_unassigned = [x == None for x in state.postman_destination_node] # A postman is idling if his current destination is None

    if if_display_details: print(if_postman_unassigned)
    if_assignment_required = [False for _ in range(NN)]  # this is for destinations
    delete_order = []

    # with respect to eah individual postman, can triger the recursion
    unassigned_node_set = list(set(state.unassigned_order_node))
    if if_display_details: print('non repeating', unassigned_node_set)
    withhold_orders = []  # indices of orders to be withheld. orders shall not be assigned because postmen are there.
    withhold_locations = []
    # -------------------------------------- Close proximity assignment----------------------------
    for unassigned_node in unassigned_node_set:
        all_indices = [index for index, element in enumerate(state.unassigned_order_node) if element == unassigned_node]
        if all_indices == []:  # check module
            print('error, empty indices list')
        # indices of all unassigned orders with this element
        all_popuptimes = [element for index, element in enumerate(state.unassigned_order_popuptime) if index in all_indices]  # find the popuptime of all customers in this region
        all_timewindow = [element for index, element in enumerate(state.unassigned_order_timewindow) if index in all_indices]
        try:
            earliest_order = all_indices[int(np.argmin(all_popuptimes))]
            # find the earliest customer, only assign this one, others put aside
        except:
            print(state.order_popuptime)
            print(all_indices)
            print(all_popuptimes)
            sys.exit()

        if unassigned_node not in state.postman_destination_node:  # if not assigned, then we update this value
            state.node_earliest_popuptime[unassigned_node] = min(all_popuptimes)
            state.node_earliest_timewindow[unassigned_node] = all_timewindow[int(np.argmin(all_popuptimes))]

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
                 #   print(earliest_order)
                  #  print(state.unassigned_order_node)
                   # print(state.unassigned_order_popuptime)
                    #print(state.unassigned_order_staytime)
                    state.postman_destination_node[p_idx] = int(state.unassigned_order_node[earliest_order])
                    state.postman_destination_indice[p_idx] = int(state.unassigned_order_indice[earliest_order])
                    if_postman_unassigned[p_idx] = False
                    delete_order.append(earliest_order)  # delete already assigned
                    state.postman_stayingtime[p_idx] = state.unassigned_order_staytime[earliest_order]
                    assignment_time = max(state.current_time, state.unassigned_order_popuptime[earliest_order])
                    state.elapse = min(state.elapse, state.unassigned_order_staytime[earliest_order])
                    g.add_assignment(p_idx, int(state.unassigned_order_node[earliest_order]),
                                     assignment_time, distance, state.unassigned_order_popuptime[earliest_order],
                                     state.unassigned_order_indice[earliest_order],
                                     state.unassigned_order_timewindow[earliest_order], 'close',
                                     state.unassigned_order_staytime[earliest_order])
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

    state.unassigned_order_node = [int(item) for idx, item in enumerate(state.unassigned_order_node) if idx not in delete_order]
    state.unassigned_order_indice = [int(item) for idx, item in enumerate(state.unassigned_order_indice) if idx not in delete_order]
    state.unassigned_order_popuptime = [int(item) for idx, item in enumerate(state.unassigned_order_popuptime) if idx not in delete_order]
    state.unassigned_order_timewindow = [int(item) for idx, item in enumerate(state.unassigned_order_timewindow) if idx not in delete_order]
    state.unassigned_order_staytime = [int(item) for idx, item in enumerate(state.unassigned_order_staytime) if idx not in delete_order]

    NN_unassigned_orders = []  # list of all unassigned orders sent to the neural network
    if if_display_details:
        print('withholdloc', withhold_locations)
    for i, node in enumerate(state.unassigned_order_node):
        if node in state.postman_destination_node or node in withhold_locations:  # part of the destinations
            withhold_orders.append(i)
        else:  # only those destinations with no postman heading towards shall be assigned a new postman. other orders should withhold.
            NN_unassigned_orders.append(i)  # unassigned order index
    if if_display_details:
        print('NN_unassigned_ordersindices', NN_unassigned_orders)
        print('NN_unassigned_orders', [state.unassigned_order_node[i] for i in NN_unassigned_orders])
    # -------------------------------------end of proximity assignment----------------------------------

    # -----------------------------preprocess NN assignment and greedy assignment-----------------------
    num_vertices = g.count_vertices()
    x_edges, x_edges_values = g.get_adjacency()  # adjaency matrix, edge distances
    # x_edges in a form of 0 or 1, x_edges_values are in a form of distance
    # if_order = [0 for _ in range(num_vertices)]  # if this customer destination requires assignment
    postman_feature = torch.empty(1, PN, postman_dim).to(device)
    # postman feature: current location coordinates, which node it is heading toward
    # it is currently in between which sets of coordinates. already traversed percentage
    for i in range(PN):  # assignment is also made during the initial stage
        postman_feature[0, i, 0] = postmanx[i]  # x coordinate
        postman_feature[0, i, 1] = postmany[i]  # y coordinate
        postman_feature[0, i, 4] = postman_status[i]  # idling or working (either servering or on the way) 0 for idling
        postman_feature[0, i, 5] = state.current_time / totaltime  # current time
        postman_feature[0, i, 6] = state.exp_travel_time[i] / 10  # expected travel time
        if postman_status[i] == 1:  # it is working
            if state.postman_destination_node[i] != None:
                postman_feature[0, i, 2] = xcoords[state.postman_destination_node[i]]  # destination coordinates
                postman_feature[0, i, 3] = ycoords[state.postman_destination_node[i]]  # destination coordinates
            else:
                postman_feature[0, i, 2] = postmanx[i]  # x coordinate
                postman_feature[0, i, 3] = postmany[i]  # y coordinate
        else:
            postman_feature[0, i, 2] = postmanx[i]
            postman_feature[0, i, 3] = postmany[i]
            if state.postman_destination_node[i] != None:
                print('wrong, destination does not equal to None')
                sys.exit()

    node_postman=[None for i in range(NN)]
    x_nodes_coord = torch.zeros(1, NN, node_dim).to(device)##  在这里将每个节点未来的状态信息加入。
    x_nodes_coord[0, :, 0] = torch.tensor(xcoords).to(device)  # x coordinates
    x_nodes_coord[0, :, 1] = torch.tensor(ycoords).to(device)  # y coordinate



    df = pd.DataFrame({'o_idx':list(range(len(state.unassigned_order_node))),  # o_id:  order index in the unassignedorder list
                       'n_id': state.unassigned_order_node,  # node id of unassigned order
                       'pop_t': state.unassigned_order_popuptime})# popup time of unassigned order
    earliest_order_indices = {n_id: group.sort_values(by='pop_t')['o_idx'].tolist()[0] for n_id, group in df.groupby(by='n_id')}

    for node in state.unassigned_order_node:  # find the latest appearance time
        if node not in state.postman_destination_node and node not in withhold_locations:
            if_assignment_required[int(node)] = True
        x_nodes_coord[0, node, 2] += 1  # add one unassigned order # if no customer leave it as zero.

    for i in range(NN):#delivered orders will also be counted here.
        x_nodes_coord[0, i, 3] = (state.current_time - state.node_earliest_popuptime[i] - state.node_earliest_timewindow[i]) / 10  #important,
    for i in range(PN):
        destin = state.postman_destination_node[i]
        if destin != None:
            x_nodes_coord[0, destin, 4] += 1  # if it is assigned then it becomes one
            node_postman[destin]=i #可能会覆盖，todo:之后可以改进
           # x_nodes_coord[0, destin, 5] = i #embedding(torch.LongTensor([i])[0]).to(device)

    # todo: add the forcast of each node here
    if is_predict:
        predict = state.predict_order_volume(NN, is_predict)
        predict = torch.from_numpy(predict).to(device)
        predict = predict.unsqueeze(0)
        # print('predict shape:', predict.shape)
        x_nodes_coord = torch.cat([x_nodes_coord, predict], dim = 2)
        # print('state predict:', predict)
        # print(x_nodes_coord.shape)

    all_adjacency = torch.tensor(np.identity(NN + PN)).unsqueeze(dim=0).to(device)# 图神经网络需要的输入
    all_adjacency[0, 0:NN, 0:NN] = torch.tensor(x_edges).to(device)
    allOD = torch.tensor(np.zeros([1, PN + NN, PN + NN])).to(device)# 边的权重
    allOD[0, 0:NN, 0:NN] = torch.tensor(x_edges_values).to(device)
    for i in range(PN):
        for j in range(NN):  # problem: cannot tell distance of zero and particular distances
            if state.all_edges != [] and state.all_edges[i] != None and j in state.all_edges[i] and \
                    state.postman_prev_node[i] != state.postman_destination_node[i]:  # if this node is one of the two edge nodes
                all_adjacency[0, NN + i, j] = 1
                all_adjacency[0, j, NN + i] = 1
                if j == state.all_edges[i][0]:
                    allOD[0, NN + i, j] = postman_percentage[i] * x_edges[state.all_edges[i][0], state.all_edges[i][1]]
                    allOD[0, j, NN+i] = postman_percentage[i] * x_edges[state.all_edges[i][0], state.all_edges[i][1]]
                else:
                    allOD[0, NN + i, j] = (1 - postman_percentage[i]) * x_edges[state.all_edges[i][0], state.all_edges[i][1]]
                    allOD[0, j, NN+i]=(1 - postman_percentage[i]) * x_edges[state.all_edges[i][0], state.all_edges[i][1]]
            if j == state.postman_prev_node[i] and (state.postman_destination_node[i] == None or j == state.postman_destination_node[i]):
                # right at the location
                all_adjacency[0, NN + i, j] = 1
                all_adjacency[0, j, NN + i] = 1
                allOD[0, NN + i, j] = 0
    # -----------------------------------------Greedy assignment--------------------------------------------------
    locations_to_delete = []
    if if_display_details:
        print('unassigned locs',sum(if_assignment_required))
        print('unassigned postman',sum(if_postman_unassigned))
    # three scenarios: 1. initial 2. only one unassigned postman, one assignment required. 3. more than one postmen,  multiple assignments
    # 已经有快递员的node，就不管（即不进行分配）；一对一，也就是一个快递，一个订单，直接匹配；多个快递员，一个订单，那么找最近的快递员分配给这个订单； 地图上只剩最后一个顾客，这种情况要单独处理；
    #print(sum(if_assignment_required), sum(if_postman_unassigned))
    #print()
    if sum(if_assignment_required) == 1 and sum(if_postman_unassigned) == 1 and len(state.order_node) != 1: #做一对一的匹配
        assigned_index = NN_unassigned_orders[0]
        assigned_location = state.unassigned_order_node[assigned_index]
        postman = if_postman_unassigned.index(True)
        distance1, _ = g.find_distance(start=state.postman_prev_node[postman], end=assigned_location)
        edge_distances=g.get_edges()
        all_edge_distances=[edge_distances[j][k] for j in edge_distances.keys() for k in edge_distances[j].keys()]
        if distance1<=max(all_edge_distances)*1:
            state.postman_destination_node[postman] = assigned_location
            state.postman_destination_indice[postman] = state.unassigned_order_indice[assigned_index]
            assignment_time = max(state.unassigned_order_popuptime[assigned_index], state.current_time)
            g.add_assignment(postman, int(state.unassigned_order_node[assigned_index]),
                             assignment_time, distance1, state.unassigned_order_popuptime[assigned_index],
                             state.unassigned_order_indice[assigned_index], state.unassigned_order_timewindow[assigned_index], 'greedy1'
                             , state.unassigned_order_staytime[assigned_index])
            state.postman_current_path_distance[postman] = distance1
            state.postman_stayingtime[postman] = state.unassigned_order_staytime[assigned_index]
            state.postman_assignment_time[postman] = assignment_time
            state.elapse=min(state.elapse, state.unassigned_order_staytime[assigned_index] + distance1 / speed)
            delete_order.append(assigned_index)
            locations_to_delete.append(assigned_index)
            if if_display_details:
                print('--greedyheuristicsonecustomer,assignpostman', postman, 'to destination', assigned_location,
                      'starting from', state.postman_prev_node[postman], 'at', assignment_time, 'popup', state.unassigned_order_popuptime[assigned_index]) #
    #elif initial or sum(if_assignment_required) < sum(if_postman_unassigned):  # only triggered if initial step/all postmen unassigned. otherwise not triggered.
    #    print('customers',sum(if_assignment_required),'postman',sum(if_postman_unassigned))， 运力充足的情况下；
    else:
        postman_indices = [i for i in range(len(state.postman_destination_node)) if state.postman_destination_node[i] == None]
        idling_postmennum = sum([1 for i in state.postman_destination_node if i == None])  # number of idling postmen
        OD = np.zeros([idling_postmennum, len(NN_unassigned_orders)])  # OD matrix for idling postmen and unassigned orders
        # allpathes = {}#之后没有用到
        counter = 0
        for i in NN_unassigned_orders:# 构造OD矩阵，行是快递员，列是终点，这里是greedy策略，不是图神经网络
            for j in range(idling_postmennum):  # here are the indices of idling postmen
                postmanindex = postman_indices[j]
                distance1, _ = g.find_distance(start=state.postman_prev_node[int(postmanindex)], end=state.unassigned_order_node[i])
                OD[j, counter] = distance1  # evaluate the distance from this postman to every destination this is euclidean distance and
                # allpathes[j, counter] = path
            counter += 1
        assignment, selected_postmen = Greedyoptimiser(OD, postman_indices, len(NN_unassigned_orders))
        inner_counter = 0  # reflects the index of postman in the OD matrix
        delete_order = []
        for i in assignment.keys():  # value corresponds to OD location index,key corresponds to postman
            assigned_index = NN_unassigned_orders[assignment[i]]  # OD index also the index of unassignedorder
            assigned_location = int(state.unassigned_order_node[assigned_index])
            if if_display_details:
                print('greedy assigned postman', i, 'to', assigned_location,'distance', OD[selected_postmen[inner_counter], assignment[i]]
                      ,'staying', state.unassigned_order_staytime[assigned_index], 'time', state.current_time)
            state.postman_destination_node[int(i)] = assigned_location  # assign destination
            state.postman_destination_indice[int(i)] = state.unassigned_order_indice[assigned_index]
            locations_to_delete.append(assigned_index)
            assignment_time = max(state.unassigned_order_popuptime[assigned_index], state.current_time)
            g.add_assignment(int(i),
                             int(state.unassigned_order_node[assigned_index]),
                             assignment_time,
                             OD[selected_postmen[inner_counter], assignment[i]],
                             state.unassigned_order_popuptime[assigned_index],
                             state.unassigned_order_indice[assigned_index],
                             state.unassigned_order_timewindow[assigned_index], 'greedy',
                             state.unassigned_order_staytime[assigned_index])
            state.postman_current_path_distance[i] = OD[selected_postmen[inner_counter], assignment[i]]
            state.postman_stayingtime[i] = state.unassigned_order_staytime[assigned_index]
            state.postman_assignment_time[i] = assignment_time
            state.elapse=min(state.elapse, OD[selected_postmen[inner_counter], assignment[i]] / speed + state.unassigned_order_staytime[assigned_index])
            delete_order.append(assigned_index)
            inner_counter += 1
    ## -----------------------------------------End of greedy assignment-------------------------------------------
    # -------------------------------------------Neural Network---------------------------------------------------
    '''else: # 除了以上的情况，都用神经网络来做
        NNassignment = {}
        if sum(if_postman_unassigned) > 0 and sum(if_assignment_required) > 0 and not initial:
            state.RL_decisions += 1
            NNassignment1, NNlogprobs, Entropy, state.prevemb = \
                neuralnetwork(all_adjacency, allOD, x_nodes_coord, postman_feature, if_assignment_required,  #需要告诉model哪些是有用的，图神经网络会被每一个节点学习embedding，但是不是所有的节点嵌入都有用
                              if_postman_unassigned, state.prevemb,  #这里的state prevemb现在没有用到
                              state.postman_prev_node, g, node_postman,  # posmaninitialnode：todo：应该是快递员上一次的位置；
                              NNassignment=None, NNlogprobs=None, Entropy=None, recursion=False, istraining=istraining)
            if if_display_details:
                print(NNassignment1)
                print(NNassignment1[0].items())
            for i in list(NNassignment1[0]):
                NNassignment[i] = NNassignment1[0][i]
            if batch!=None:
                buffer.logprobs[batch].append(torch.clamp(NNlogprobs[0], -100, -0.01).to(device))
                buffer.entropy[batch].append(Entropy[0].to(device))# 为什么要用熵，熵和 policy的收敛是反着的。 学习的时候，也有entropy的约束， 可以作为一个reward。
            for i in NNassignment1[0].values():
                if_assignment_required[i] = False
            allpostmen = NNassignment.keys()
            # NNassignment:dictionary left hand side postman index, righthand side is the destination index
            # list of postman
            # location numbers not indices
            for i in allpostmen:  # NNassignment[i] is the destination
                state.postman_destination_node[i] = NNassignment[i]
                # evaluate the expected arrival time. Then if change elapse is needed, go backwards in time
                D, _ = g.find_distance(state.postman_prev_node[i], NNassignment[i])
                duration = D / speed
                all_location_indices = [j for j in range(len(state.unassigned_order_node)) if
                                      state.unassigned_order_node[j] == NNassignment[i]]
                all_corresponding_popuptimes = [state.unassigned_order_popuptime[j] for j in all_location_indices]
                assigned_index=all_location_indices[np.argmin(all_corresponding_popuptimes)]
                locations_to_delete.append(assigned_index)
                assignment_time = max(state.current_time, state.node_earliest_popuptime[NNassignment[i]])
                if state.node_earliest_popuptime[NNassignment[i]]>state.current_time: #check module
                    print('error, future order pop up')
                    sys.exit()
                g.add_assignment(i, int(NNassignment[i]), assignment_time, D,
                                 state.node_earliest_popuptime[NNassignment[i]],
                                 state.unassigned_order_indice[earliest_order_indices[int(NNassignment[i])]],
                                 state.unassigned_order_timewindow[earliest_order_indices[int(NNassignment[i])]], 'RL',
                                 state.unassigned_staying_time[earliest_order_indices[int(NNassignment[i])]])
                state.postman_destination_indice[i] = state.unassigned_order_indice[
                    earliest_order_indices[int(NNassignment[i])]]
                state.postman_current_path_distance[i] = D
                state.postman_stayingtime[i] = state.unassigned_staying_time[earliest_order_indices[int(NNassignment[i])]]
                state.postman_assignment_time[i] = assignment_time
                state.elapse=min(state.elapse, duration + state.unassigned_staying_time[earliest_order_indices[int(NNassignment[i])]])
                #-------------------------------assigned index is the unassigned postman index----------------------------
                if if_display_details:
                    print('nnassignment', NNassignment)
                    print('--neuralnetwork,assignpostman', i, 'to destination', NNassignment[i], 'starting from',
                          state.postman_prev_node[i], 'at', assignment_time, 'popup', state.node_earliest_popuptime[NNassignment[i]],
                          'exptraduration', duration,'staying', state.unassigned_staying_time[earliest_order_indices[int(NNassignment[i])]])
                if state.postman_prev_node[i] == NNassignment[i]:
                    print('samenode not detected')
                    sys.exit()'''
    # -------------------temporary exit command--------------------------
    # check if there are duplicates in a list
    a_set = set(locations_to_delete)
    contains_duplicates = len(locations_to_delete) != len(a_set)
    if contains_duplicates:
        print('contains duplicates')
        sys.exit()
    state.unassigned_order_node = [state.unassigned_order_node[i] for i in range(len(state.unassigned_order_node)) if i not in locations_to_delete]
    state.unassigned_order_indice = [state.unassigned_order_indice[i] for i in range(len(state.unassigned_order_indice)) if i not in locations_to_delete]
    state.unassigned_order_popuptime = [state.unassigned_order_popuptime[i] for i in range(len(state.unassigned_order_popuptime)) if i not in locations_to_delete]
    state.unassigned_order_timewindow = [state.unassigned_order_timewindow[i] for i in range(len(state.unassigned_order_timewindow)) if i not in locations_to_delete]
    state.unassigned_order_staytime = [state.unassigned_order_staytime[i] for i in range(len(state.unassigned_order_staytime)) if i not in locations_to_delete]
    state.current_time = state.current_time + state.elapse
    if state.elapse == totaltime:
        print('error, over elapse')
        print(state.order_node)
        print(state.order_popuptime)
        print(state.postman_destination_node)
        sys.exit()
    return g, state


def caluate_and_log_reward(g: Env.Graph, log_config: dict):
    """
    calcuate the reward, and then save the details into the desk
    :param g:
    :param log_config: {"fout": str, "sheet_name": str, "server_version":bool, "time":float, 'rl_decision_num': int}
    if log_config['fout'] == 'not_save', then we don't save the details
    :return:
    """
    postman_assigned_nodes, postman_assigned_time, postman_assigned_distance, postman_assigned_popuptime, \
    postman_assigned_index, postman_assigned_timewindow, assignment_method, postman_assigned_staytime = g.get_assigned()
    # self.postman_delivered_nodes, self.postman_delivered_times, self.postman_delivered_index
    postman_delivered_nodes, postman_delivered_times, postman_delivered_index = g.get_delivered()
    for i in range(PN):
        postman_assigned_nodes[i] = sort_together([postman_assigned_index[i], postman_assigned_nodes[i]])[1]
        postman_assigned_time[i] = sort_together([postman_assigned_index[i], postman_assigned_time[i]])[1]
        postman_assigned_popuptime[i] = sort_together([postman_assigned_index[i], postman_assigned_popuptime[i]])[1]
        postman_assigned_distance[i] = sort_together([postman_assigned_index[i], postman_assigned_distance[i]])[1]
        assignment_method[i] = sort_together([postman_assigned_index[i], assignment_method[i]])[1]
        postman_delivered_times[i] = sort_together([postman_assigned_index[i], postman_delivered_times[i]])[1]
        postman_assigned_timewindow[i] = sort_together([postman_assigned_index[i], postman_assigned_timewindow[i]])[1]
        postman_assigned_staytime[i] = sort_together([postman_assigned_index[i], postman_assigned_staytime[i]])[1]
        postman_delivered_index[i] = sort_together([postman_assigned_index[i], postman_delivered_index[i]])[1]
        postman_assigned_index[i] = sort_together([postman_assigned_index[i], postman_assigned_index[i]])[1]
    episode_reward = Env.episode_reward(popup_times=postman_assigned_popuptime,
                                        postman_assigned_index=postman_assigned_index,
                                        delivered_times=postman_delivered_times,
                                        postman_delivered_index=postman_delivered_index,
                                        time_windows=postman_assigned_timewindow,
                                        staying_times=postman_assigned_staytime,
                                        assigned_distance=postman_assigned_distance)

    if log_config['fout'] == 'not_save': return episode_reward # directly return the reward without saving the details
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
    write_xlsx(df=df, fout=log_config['fout'], sheet_name=log_config['sheet_name'], server_version=log_config['server_version'])
    if len(postman_assigned_popuptime) != len(postman_delivered_times) or len(postman_delivered_nodes) != len(postman_delivered_times):
        print('error, wrong size assigned and delivered tasks')
        print('postmanassignedtasks', len(postman_assigned_nodes), 'postmandeliveredtasks', len(postman_delivered_nodes))
        sys.exit()
    return episode_reward

# Set up lists to hold results
class Exp_Buffer():
    def __init__(self, buffersize):
        self.total_rewards = []
        self.batch_rewards = []  # average reward along this batch
        self.batch_actions = []
        self.batch_states = []
        self.batch_graphs = []
        self.logprobs = [[] for _ in range(buffersize)]
        self.batch_counter = 0
        self.state_counter = 0
        self.entropy = [[] for _ in range(buffersize)]
    
    def clear_logprobs(self):
        self.logprobs =[[] for _ in range(buffersize)]
        self.entropy = [[] for _ in range(buffersize)]
        
    
    def clear_rewards(self):
        self.batch_rewards = []
        self.total_rewards = []
        self.batch_counter = 0
    
    def add_new_instance(self, action, reward):
        self.total_rewards.append(reward)
        self.batch_actions.append(action)
        self.batch_counter += 1
        if self.batch_counter > buffersize:
            del self.total_rewards[0]
            del self.batch_actions[0]
            self.batch_counter -= 1
    
    def add_new_state(self, state, graph):
        self.batch_states = state
        self.batch_graphs = graph
        self.state_counter += 1
        if self.state_counter > 1:  # can only store one in the buffer
            self.state_counter -= 1
    
    def add_batch_rewards(self, averagereward, num):
        self.batch_rewards = self.batch_rewards + [averagereward for _ in range(num)]
    
    def get_instance(self):
        return self.batch_states, self.batch_graphs, self.total_rewards, self.batch_rewards, self.logprobs

actor_net = NetModel.ResidualGatedGCNModel(is_training, PNlower)
actor_net.to(device)


# Define optimizer
optimizer = optim.Adam(actor_net.parameters(), lr=learningrate)
MseLoss = nn.MSELoss() #为什么用MSE作为loss
buffer = Exp_Buffer(buffersize)
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

if is_training==False:
    num_scenarios=1
    batch_size=1
trainingsteps=0
# global greedythreshold
percentage_improvements=[]
all_losses=[]
RL_test_rewards=[]
greedy_optimiser_rewards=[]
Greedy_test_rewards=[]
try:
    allepisodes = flatten([[2 * i + 1000, 2 * i + 1 + 1000] for i in range(num_episodes)])
    while graph_seed < num_graphs: #num graphs：要生成多少个graph，graphseed是生成graph的随机种子
        for ep in range(allepisodes):
            #if ep % training_size==0: # including the first one
             #   greedy_rollout_rewards=[]
              #  greedy_optimiser_rewards=[]
            #for scenario in range(num_scenarios):
            RL_episode_reward = 0
            # -----------------------generate random problems-------------------------
            if ep % 2 == 0 and is_training:
                print('saving parameters')
                torch.save(actor_net.state_dict(), actordict)

            PN = PNlower if PNlower == PNhigher else np.random.randint(low=PNlower, high=PNhigher)
            NN = NNlower if NNlower == NNhigher else np.random.randint(low=NNlower, high=NNhigher) # todo: should change to poisson distribution

            g1 = Env.Graph(PN)  # initialise a graph
            g1 = Env.graph_generator(g1, NN, randomseed=graph_seed, graphname=None,Euclidean=Euclidean) #NN: number of nodes in the graph
            state1 = Env.State(PN, NN, device)  # redeclare state
            g1, state1 = Env.random_generator(g1, lb, hb, NN, PN, state1, randomseed=ep)
            state1.current_time = 0
            if display_myopic:
                g2 = copy.deepcopy(g1)
                state2 = copy.deepcopy(state1)
                # --------------------end of random graph generation---------------------------
                # ----------------------greedy initialisation----------------------------------
                starttime = time.time()
                g2, state2 = Env.greedy_policy(state2, g2, PN, initial=True)
                g2, state2 = Env.Environment(g2, state2, initial=True)
                state2.elapse = 0
                g2, state2 = Env.greedy_policy(state2, g2, PN, initial=False)
                greedy_counter = 0
                state2.Done = False
                # ------------------------end of greedy initialisation----------------------------
                # ------------------------run greedy policy---------------------------------------
                print('----------run greedy policy-----------')
                # greedy policy does not require iterate over many parallel batches
                while not state2.Done:
                    g2, state2 = Env.Environment(g2, state2, initial=False)
                    if state2.Done: break#@? 怎么样才算是状态结束？
                    state2.update_postman_xy(g2)#更新快递员位置坐标
                    g2, state2 = Env.greedy_policy(state2, g2, PN, initial=False)
                    greedy_counter += 1
                log_cfg = {'fout':path + 'GreedyStatistics.xlsx', "sheet_name":f'graph_{graph_seed}', 'server_version':serverversion, 'time': time.time() - starttime}
                episode_reward,_ = caluate_and_log_reward(g2, log_cfg)

                greedy_optimiser_rewards.append(episode_reward)
                if not is_training: Greedy_test_rewards.append(episode_reward)
                print('Greedy reward', episode_reward)
                state2.reset()
                g2.reset(NN)  # reset the graph
                g2, state2 = Env.random_generator(g2, lb, hb, NN, PN, state2, randomseed=ep)
                # --------------------------end of greedy policy---------------------------------
            #-----------------------------start of greedy RL rollout initialisation-----------------------------
            if is_training:
                print('run greedy rollout rl')
                g3 = copy.deepcopy(g1)
                state3 = copy.deepcopy(state1)
                g3, state3 = RLpolicy(g3,PN,state3, None,0, initial=True,neuralnetwork=actor_net_target, istraining=False)
                g3, state3 = Env.Environment(g3, state3, initial=True)
                state3.elapse = 0
                g3, state3 =RLpolicy(g3,PN,state3, None,1,initial=False,neuralnetwork=actor_net_target, istraining=False)
                greedy_counter = 0
                state3.Done = False
                while not state3.Done:
                    g3, state3 = Env.Environment(g3, state3, initial=False)
                    if state3.Done: break
                    state3.update_postman_xy(g3)
                    g3, state3 = RLpolicy(g3, PN, state3, None, greedy_counter+2, initial=False, neuralnetwork=actor_net_target, istraining=False)
                    greedy_counter += 1
                RLgreedyepisodereward ,_= caluate_and_log_reward(g3, {'fout':"not_save"})
                RL_test_rewards.append(RLgreedyepisodereward)
                #sys.exit()
            print(sum([-(RL_test_rewards[k]-greedy_optimiser_rewards[k]) for k in range(len(RL_test_rewards))])/sum(greedy_optimiser_rewards))
            print(RL_test_rewards)
            print(greedy_optimiser_rewards)
            #-----------------------------end of greedy RL rollout---------------------------------------
            # -------------------------RL initialisation-------------------------------------
            '''print('run rl')
            buffer.add_new_state(state1, g1)
            if display_VRPTW:
                df, episode_reward = VRPTW.RecursiveReoptimisation(PN, NN, graph_seed, ep)
                write_xlsx(df=df, fout=path + 'VRPTWStatistics.xlsx', sheet_name=f'graph_{graph_seed}', server_version=serverversion)
            if display_RL:
                for batch in range(batch_size):
                    print('Episode:', ep, 'batch', batch)
                    if is_training:
                        greedy_rollout_rewards.append(RLgreedyepisodereward)
                    starttime = time.time()

                    g1, state1 = RLpolicy(g1, PN, state1, (ep % training_size) * batch_size + batch, 0, initial=True, neuralnetwork=actor_net)
                    g1, state1 = Env.Environment(g1, state1, initial=True)
                    state1.elapse = 0
                    g1, state1 = RLpolicy(g1, PN, state1, (ep % training_size) * batch_size + batch, 1, initial=False, neuralnetwork=actor_net)
                    RLcounter = 0
                    state1.Done = False
                    # -------------------------end of RL initialisation----------------------------
                    # --------------------------start of RL policy----------------------------------
                    while state1.Done != True:
                        g1, state1 = Env.Environment(g1, state1, initial=False)
                        if state1.Done: break
                        state1.update_postman_xy(g1)
                        g1, state1 = RLpolicy(g1, PN, state1, (ep%training_size) * batch_size + batch, RLcounter + 2, initial=False,
                                              neuralnetwork=actor_net)
                        RLcounter += 1

                    log_cfg = {'fout':path+'RLStatistics.xlsx', 'sheet_name':f'graph_{graph_seed}',
                               'server_version':serverversion, 'time': time.time() - starttime, 'rl_decision_num':state1.RL_decisions}
                    episode_reward = caluate_and_log_reward(g1, log_cfg)
                    if not is_training:
                        RL_test_rewards.append(episode_reward)
                    buffer.add_new_instance(state1.RL_decisions, episode_reward)
                    RL_episode_reward += episode_reward

                    state1.reset()
                    g1.reset(NN)  # reset the graph
                    g1, state1 = Env.random_generator(g1, lb, hb, NN, PN, state1, randomseed=ep)'''
            # ------------------------------end of RL policy--------------------------------
            # ----------------------------batch counter finishes----------------------------
            # ------------------------------update policy-----------------------------------
            '''if PPO == False and is_training and (ep+1) % training_size==0:
                print('Enter Training')
                print('episode', ep)
                _, _, old_total_rewards, _, old_logprobs = buffer.get_instance()
                logprob_len=[len(old_logprobs[i]) for i in range(len(old_logprobs))]
                print('logprob len:', logprob_len)
                greedy_optimiser_rewards=[greedy_optimiser_rewards[i] for i in range(len(greedy_optimiser_rewards)) for j in range(batch_size)]
               # print(greedy_optimiser_rewards)
                selected=[i for i in range(len(logprob_len)) if logprob_len[i] > 0]
                greedy_optimiser_rewards = [greedy_optimiser_rewards[i] for i in selected]
                greedyrolloutreward = torch.tensor(greedy_rollout_rewards)[selected].to(device)
                reward_tensor = torch.FloatTensor(old_total_rewards)[selected].to(device)
                oldlogprobs=torch.stack([torch.stack(old_logprobs[i]).sum() for i in range(len(old_logprobs)) if i in selected])
                selected_logprobs = (reward_tensor - greedyrolloutreward) * oldlogprobs  # -0.1*torch.stack(buffer.entropy,dim1).sum(dim=1)
                #print('avgreward',reward_tensor.view(training_size,-1).mean(dim=-1))
                print('reward_tensor:', reward_tensor)
                print('greedy rollout', greedy_rollout_rewards)
                percentage_improvement= (sum(greedy_optimiser_rewards) - sum(greedyrolloutreward.tolist())) / sum(greedy_optimiser_rewards)
               
                percentage_improvements.append(percentage_improvement)
                loss = selected_logprobs.mean()  # direction should be along the positive size. The loss is supposed to be large negative
                print('percentage improvement', percentage_improvement)
                print('loss',loss)
                optimizer.zero_grad()
                # Calculate gradients
                all_losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_net.parameters(), 2, norm_type=2)
                # Apply gradients
                optimizer.step()
                # ------------------------------update counter----------------------------------
                buffer.clear_logprobs()
                buffer.clear_rewards()
                if trainingsteps%baseline_update_steps and trainingsteps!=0:
                    actor_net_target=copy.deepcopy(actor_net)
                    print('update actor net target')
                trainingsteps+=1'''
                
        graph_seed += 1

except KeyboardInterrupt:
    print('saving parameters from keyboard interruption')

torch.save(actor_net.state_dict(), actordict)
if is_training:
    loss_records=pd.DataFrame({'all_losses':all_losses,'percentage improvement':percentage_improvements})
    loss_records.to_csv(path+'loss.csv')
else:
    if len(Greedy_test_rewards)>len(RL_test_rewards):
        del Greedy_test_rewards[-1]
    elif len(Greedy_test_rewards)<len(RL_test_rewards):
        del RL_test_rewards[-1]
    print(len(Greedy_test_rewards))
    print(len(RL_test_rewards))
    test_results=pd.DataFrame({'greedy':Greedy_test_rewards,'RL':RL_test_rewards})
    test_results.to_csv(path+'test_results.csv')
print('properly saved')
