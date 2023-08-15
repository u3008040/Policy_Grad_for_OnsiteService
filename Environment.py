
# -*- coding: utf-8 -*-
import numpy as np
import sys
import copy
import torch

from my_utils import GraphKruskal, dijkstra, is_all_none, delete_by_idx
from GreedyOptimisation import Greedyoptimiser
import utils.Parameters as Parameters


args = Parameters.parameters()
global speed, unittime, totaltime, lowerbound, higherbound, low_stay_bound, high_stay_bound, INnumber, CPnumber, predict_time_grain
global TWlb, TWub
displayoutlet=0
speed = vars(args)['speed']
totaltime = vars(args)['totaltime']
lb = vars(args)['low_order_bound']
hb = vars(args)['high_order_bound']
TWlb=vars(args)['TWlb']
TWub=vars(args)['TWub']
low_stay_bound = vars(args)['low_stay_bound']
high_stay_bound = vars(args)['high_stay_bound']
INnumber = vars(args)['INnumber']
CPnumber = vars(args)['CPnumber']
NNlower=vars(args)['NNlower']
NNhigher=vars(args)['NNhigher']
PNlower=vars(args)['PNlower']
PNhigher=vars(args)['PNhigher']
hidden_dim=vars(args)['hidden_dim']
predict_time_grain = vars(args)['predict_time_grain']

# note: if inconsistent tab and space, simply click on control alt + L to clean up the code


class Graph:
    def __init__(self, PN, graphname=None):  # memorise a graph, graph has current customer and current postmen. connectivity
        #self.postmans = [Postman() for _ in range(PN)]
        self.graphname = graphname
        self.xcoord = []
        self.ycoord = []
        self.postman_x = {}
        self.postman_y = {}
        self.postman_status={}
        self.postman_percentage={}
        self.current_order_node = []
        self.connected_to = {}
        self.counter = 0
        self.PN=PN
        self.postman_node = [None for _ in range(PN)]
        self.postman_delivered_index=[[] for _ in range(PN)]
        self.postman_delivered_nodes = [[] for _ in range(PN)]
        self.postman_delivered_times = [[] for _ in range(PN)]
        self.nearest3nodes={}
        self.postman_assigned_index=[[] for _ in range(PN)]
        self.postman_assigned_nodes = [[] for _ in range(PN)]
        self.postman_assigned_time = [[] for _ in range(PN)]
        self.postman_assigned_distance=[[] for _ in range(PN)]
        self.assignment_time_order1 = [[] for _ in range(self.PN)]
        self.assignment_time_order = 0
        self.postman_assigned_timewindow= [[] for _ in range(PN)]
        self.postman_assigned_popuptime=[[] for _ in range(PN)]
        self.postman_assigned_stayingtime = [[] for _ in range(PN)]

        self.assignment_method = [[] for _ in range(PN)]
        self.adjacency=[]
        self.edge_distance=[]
        self.OD={}
        self.ODlist=[]
        self.node_orders = {}
        self.node_all_orders = {}
        self.all_pathes={}
        self.add_penalty=0
        
    def reset(self,NN):
        self.postman_x = {}
        self.postman_y = {}
        self.postman_status = {}
        self.postman_percentage = {}
        self.current_order_node = [] # 当前所有订单对应的node
        self.assignment_time_order1 = [[] for _ in range(self.PN)]
        self.assignment_time_order = 0
        self.postman_node = [[] for _ in range(self.PN)]#todo, 这些都可以用一个快递员类来描述
        self.postman_delivered_index = [[] for _ in range(self.PN)]
        self.assignment_method = [[] for _ in range(self.PN)]
        self.postman_assigned_stayingtime = [[] for _ in range(self.PN)]
        self.postman_assigned_index = [[] for _ in range(self.PN)]
        self.postman_delivered_nodes = [[] for _ in range(self.PN)]
        self.postman_delivered_times = [[] for _ in range(self.PN)]
        self.postman_assigned_nodes = [[] for _ in range(self.PN)]
        self.postman_assigned_time = [[] for _ in range(self.PN)]
        self.postman_assigned_timewindow = [[] for _ in range(self.PN)]
        self.node_orders = {i: [] for i in range(NN)} # 记录一个节点有哪些订单
        self.node_all_orders={i:0 for i in range(NN)}
        self.postman_assigned_distance = [[] for _ in range(self.PN)]
        self.postman_assigned_popuptime = [[] for _ in range(self.PN)]
        self.add_penalty=0
    
    def add_coords(self, x, y):  # add coordinates
        self.xcoord.append(x)
        self.ycoord.append(y)
        self.connected_to[self.counter] = {}
        self.node_orders[self.counter] = []
        self.node_all_orders[self.counter]=0
        self.counter += 1
        self.adjacency=np.identity(self.counter)
        self.edge_distance=np.zeros([self.counter, self.counter])
    
    def add_order_node(self, order_node):  # add an order's node  if the order pops up
        self.current_order_node.append(int(order_node))
    
    def erase_order_node(self, order_node):  # delete order's node if not needed
        self.current_order_node.remove(int(order_node))
    
    def set_postman_node(self, postman, node):  # add or change postman location.
        # postman location must be at one of the destination, it is an index
        # only triggered upon arrival or prior to the first job. The first job must be at one of the destinations
        self.postman_node[int(postman)] = int(node)
    
    def update_postman(self, postmanx, postmany, postman, status, percentage):  # add or change postman location in terms of coordinates
        if postmanx!=None:
            self.postman_x[int(postman)] = postmanx
        if postmany!=None:
            self.postman_y[int(postman)] = postmany
        self.postman_status[int(postman)]=status #status=0 means postman is idling, status=1 means postman is working/moving
        self.postman_percentage[int(postman)]=percentage
    
    def add_neighbor(self, vertex, neighbour, weight=0):  # add neighbours weight is the distance.
        self.connected_to[int(vertex)][int(neighbour)] = weight
        self.connected_to[int(neighbour)][int(vertex)] = weight
        self.adjacency[int(neighbour)][int(vertex)]=1
        self.adjacency[int(vertex)][int(neighbour)]=1
    
    def add_edge(self, f_vertex, t_vertex, weight):  # same as the previous one, a repeating step.
        self.add_neighbor(int(f_vertex), int(t_vertex), weight)
    
    def get_postman_node(self, postman):  # obtain postman's node. do not retrieve if not necessary
        return self.postman_node[int(postman)]
    
    def get_edges(self):
        return self.connected_to
    
    def get_vertices(self):
        return self.xcoord, self.ycoord
    
    def count_vertices(self):  # count how many nodes are there
        return len(self.xcoord)
    
    def get_postman(self):  # get postman x and y coordinates
        return self.postman_x, self.postman_y, self.postman_status, self.postman_percentage
    
    def get_current_order_node(self):  # get current order nodes
        return self.current_order_node
    
    def add_delivered(self, postman, task, finishingtime, orderindex): #may be inconsistent
        self.postman_delivered_nodes[postman].append(int(task))
        self.postman_delivered_times[postman].append(finishingtime)
        self.postman_delivered_index[postman].append(int(orderindex))
        self.node_orders[task].remove(int(orderindex))
      #  self.node_all_orders[task]+=1
      #  print(self.node_all_orders)
        
    def add_assignment(self, postman, task, assigntime, distance, popuptime, orderindex, timewindow, assignmethod, staytime):
        self.postman_assigned_nodes[postman].append(int(task))
        self.postman_assigned_time[postman].append(assigntime)
        self.assignment_time_order1[postman].append(self.assignment_time_order)
        self.assignment_time_order += 1
        self.postman_assigned_distance[postman].append(distance)
        self.postman_assigned_popuptime[postman].append(popuptime)
        self.postman_assigned_index[postman].append(int(orderindex))
        self.assignment_method[postman].append(assignmethod)
        self.postman_assigned_timewindow[postman].append(timewindow)
        self.postman_assigned_stayingtime[postman].append(staytime)
        self.node_orders[task].append(int(orderindex))

    def get_delivered(self, if_print=False):
        if if_print:
            print(self.postman_delivered_nodes)
            print(self.postman_delivered_times)
        return self.postman_delivered_nodes, self.postman_delivered_times, self.postman_delivered_index
    
    def get_assigned(self, if_print=False):
        if if_print:
            print(self.postman_assigned_nodes)
            print(self.postman_assigned_time)
            print(self.postman_assigned_distance)
            print(self.postman_assigned_popuptime)
        return self.postman_assigned_nodes, self.postman_assigned_time, self.postman_assigned_distance, self.postman_assigned_popuptime\
            , self.postman_assigned_index, self.postman_assigned_timewindow, self.assignment_method, self.postman_assigned_stayingtime,\
                self.assignment_time_order1
    
    def get_adjacency(self):
        return self.adjacency, self.ODlist
    
    def add_additional_penalty(self,penalty):
        self.add_penalty+=penalty
        
    def deriveOD(self):
        self.ODlist=np.zeros([len(self.xcoord),len(self.xcoord)])
        for i in range(len(self.xcoord)):
            self.OD[i]={}
            self.all_pathes[i]={}
            for j in range(i,len(self.xcoord)):
                self.OD[i][j],self.all_pathes[i][j]=\
                    dijkstra(self.get_edges(), i, j)
                self.ODlist[i,j]=self.OD[i][j]
                self.ODlist[j,i]=self.OD[i][j]
        # find the nearest 3 euclidean nodes
        for i in range(self.counter):
            distance=[]
            for j in range(self.counter):
                distance.append(((self.xcoord[i]-self.xcoord[j])**2+(self.ycoord[i]-self.ycoord[j])**2)**0.5)
            indices=np.argsort(distance)
            self.nearest3nodes[i]=np.array([self.ODlist[indices[k]][i] for k in range(1,self.counter)])
            
    def find_distance(self, start, end):
        start=int(start)
        end=int(end)
        if start==end:
            return 0, [start,start]
        elif start > end: #flip it
            return self.OD[end][start], self.all_pathes[end][start][::-1]
        elif start<end: #no need to flip
            return self.OD[start][end], self.all_pathes[start][end]
    
    def findOD(self,start,end,Euclidean=False):
        start = int(start)
        end = int(end)
        if start == end:
            return 0
        elif start > end:  # flip it
            return self.OD[end][start]
        elif start < end:  # no need to flip
            return self.OD[start][end]
        
class State:
    def __init__(self, PN, NN, device):
     # memorise the state information
        self.PN=PN
        self.NN=NN
        self.device=device
        self.elapse=0
        self.delivered_node=[]# 已经完成的订单的 node
        self.delivered_postman=[]  # 完成了任务的快递员编号
        self.postman_current_path=[] # 每个快递员当前路径
        self.postman_current_path_distance=[0 for i in range(PN)]     #目前整个路径的长度
        self.all_tra_percentage=[]
        self.prev_edge=[]
        self.order_popuptime=[]       #订单产生时间
        self.order_node=[]   # 所有订单所在节点
        self.order_indice=[]     # 所有订单编号
        self.postman_prev_node=[]   # 快递员上一个时刻所在节点
        self.order_stay_time=[]       #所有订单stay_time
        self.if_assigned=[]
        self.all_edges=[]
        self.unassigned_order_node=[]   #没有分配的订单对应的节点
        self.unassigned_order_indice=[]    #没有分配的订单对应的编号
        self.unassigned_order_popuptime=[]
        self.unassigned_order_timewindow=[]
        self.postman_destination_node=[]
        self.postman_elapse=[]
        self.node_earliest_indice={}     # 某一个node最早出现的订单index
        self.node_earliest_popuptime={}
        self.node_earliest_timewindow= {}
        self.node_latest_indice={}
        self.node_latest_popuptime={}
        self.node_latest_timewindow={}
        self.order_timewindow=[]    # 所有订单time window
        self.previous_time=0
        for i in range(NN):
            self.node_earliest_timewindow[i]=0
            self.node_earliest_popuptime[i]=0
            self.node_latest_popuptime[i]=0
            self.node_latest_timewindow[i]=0
        self.current_time=0
        self.postman_xy=[]   #快递员位置坐标
        self.change_elapse = []
        self.unassigned_order_staytime=[]
        self.postman_destination_indice=[] # 当前快递员目的订单编号
        self.postman_assignment_time=[0 for _ in range(PN)]   #每个人最近一次派单时间
        self.postman_stayingtime=[0 for _ in range(PN)]
        self.already_stayed=[0 for _ in range(PN)]  #快递员已经到了终点， 并且停留了多久
        self.exp_travel_time=[0 for _ in range(PN)]
        self.prevemb=torch.zeros(1,NN,hidden_dim).to(device)
  #which job it is assigned by index provided by locations or popuptime, length is the number of postmen
        self.Done=False
        self.RL_decisions=0
        
    def reset(self):
        self.elapse = 0
        self.delivered_node = []
        self.delivered_postman = []
        self.postman_current_path = []
        self.order_indice = []
        self.all_tra_percentage = []
        self.prev_edge = []
        self.order_popuptime = []
        self.order_node = []
        self.postman_prev_node = []
        self.order_stay_time = []
        self.if_assigned = []
        self.all_edges = []
        self.unassigned_order_node = []
        self.unassigned_order_indice=[]
        self.unassigned_order_popuptime = []
        self.unassigned_order_timewindow = []
        self.postman_destination_node = [] #快递员目标位置
        self.postman_destination_indice = [] #快递员目标订单
        self.postman_elapse = []
        self.previous_time = 0
        self.node_earliest_indice = {}
        self.node_earliest_popuptime = {}
        self.node_earliest_timewindow={} #the timewindow associated with the current earliest customer
        self.order_timewindow = []
        for i in range(self.NN):
            self.node_earliest_timewindow[i] = 0
            self.node_earliest_popuptime[i] = 0
        self.current_time = 0
        self.postman_xy = []
        self.change_elapse = []
        # self.assignmentstayingtime = []
        self.unassigned_order_staytime = []
        self.postman_assignment_time = [0 for _ in range(self.PN)]
        self.postman_stayingtime = [0 for _ in range(self.PN)]
        self.already_stayed = [0 for _ in range(self.PN)]
        self.exp_travel_time = [0 for _ in range(self.PN)]
        self.prevemb = torch.zeros(1,self.NN, hidden_dim).to(self.device)
        # which job it is assigned by index provided by locations or popuptime, length is the number of postmen
        self.Done = False
        self.RL_decisions=0

    def update_postman_xy(self, g: Graph):
        self.postman_xy = []  # 快递员位置坐标
        postmanx, postmany, _, _ = g.get_postman()
        for i in range(len(self.postman_prev_node)):
            self.postman_xy.append([postmanx[i], postmany[i]])

    def predict_order_volume(self, NN: int, is_predict: bool):
        global predict_time_grain
        predcit_T = int(totaltime / predict_time_grain)

        ON = np.random.randint(low=lb, high=hb, size=1)
        assert totaltime % predict_time_grain == 0, f'wrong in total time: {totaltime} and predict time grain:{predict_time_grain}.'
        poptime_predit_grain = [int(t) for t in  get_random(min_v=0, max_v=predcit_T - 1, n=ON)]  # 预测的时候用到
        order_node = [int(_) for _ in get_random(min_v=0, max_v=NN - 1, n=ON)]  # 预测的时候用到

        predict =  np.zeros((NN, predcit_T))
        if not is_predict: return predict #不预测就直接返回空值
        for node_idx, time_idx in zip(order_node, poptime_predit_grain):
            predict[node_idx][time_idx] += 1
        return predict



def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# NN number of nodes in this map
# this generator generate a map at a time
def graph_generator(g, NN,Mag_Factor, lb=7, ub=10, randomseed=None, graphname=None,Euclidean=False):
    # randomly generate a graph. with random number of nodes and connectivity
    # lb and ub are lower and upper bounds of nodal connectivity, here is different from random generator
    if randomseed != None:
        np.random.seed(randomseed)
    Nodes = np.random.rand(NN, 2)*Mag_Factor  # two coordinates
    Nodeconnectivity = np.random.randint(lb, ub, size=NN)  # how many nodes to connect to others
    
    if Euclidean==False:
        EucOD = np.empty([NN, NN])  # create an euclidean OD as well, but it will not be used as the true OD matrix
        gKruskal = GraphKruskal(NN)  # use Kruskal algorithm to generate a minimal spanning tree. so that
        # no single node is standalone
        for i in range(NN):
            EucOD[i, i] = 0
            g.add_coords(Nodes[i, 0], Nodes[i, 1])
            for j in range(i, NN):
                length = ((Nodes[i, 0] - Nodes[j, 0]) ** 2 + (Nodes[i, 1] - Nodes[j, 1]) ** 2) ** 0.5
                EucOD[i, j] = length
                EucOD[j, i] = length
                gKruskal.addEdge(i, j, EucOD[i, j])
        string, minimumCost, assignments, comparisons = gKruskal.Kruskal()
        for i in range(NN):
            for j in range(min(Nodeconnectivity[i], NN - 1)):
                trytoconnect = np.argsort(EucOD[i, :])[j + 1]  # derive the index of this node
                Failure = False
                for k in range(len(string)):  # if it intersects with a particular node, then mark it as fail
                    if intersect(Nodes[i], Nodes[trytoconnect], Nodes[string[k][0]], Nodes[string[k][1]]):
                        Failure = True
                if not Failure:  # if not failure
                    if [i, trytoconnect] not in string:
                        string.append([i, trytoconnect])
        for i in range(len(string)):  # connect edges
            g.add_edge(string[i][0], string[i][1], EucOD[string[i][0], string[i][1]] * (1 + np.random.rand() * 0.2))
    else:
        for i in range(NN):
            g.add_coords(Nodes[i, 0], Nodes[i, 1])
        for i in range(NN):
            for j in range(i,NN):
                g.add_edge(i,j,((Nodes[i, 0] - Nodes[j, 0]) ** 2 + (Nodes[i, 1] - Nodes[j, 1]) ** 2) ** 0.5)
                
    # eliminate subgraphs
    # check with shortest path algorithms
    # then try to connect the unconnected one.
    g.deriveOD()
    if not Euclidean: del EucOD
    np.random.seed()
    
    return g


def get_random(min_v, max_v, n):
    "generate n data in [min_v, max_v] that obeys the normal distribution"
    mu = (max_v - min_v) / 2
    sigma = (max_v - mu) / 3
    x = np.random.normal(mu, sigma, n)
    res = []
    for _ in x:
        _ = max(_, min_v)# 数据范围控制
        _ = min(_, max_v)
        res.append(_)
    return res

def random_generator(g, lb, ub, lam, NN, PN, state, randomseed=None):  # PN is the postman number, NN is the number of nodes
    # here is upper and lower bounds are for ON, ON is the order number in the entire time span
    state = copy.deepcopy(state)
    if randomseed != None: np.random.seed(randomseed)
    intervals = [0]
    sum_intervals = 0
    max_duration = totaltime
    while sum_intervals < max_duration:
        interval = np.random.exponential(1 / lam)
        sum_intervals += interval
        if sum_intervals < max_duration:
            intervals.append(interval)
        else:
            intervals.append(max_duration - sum(intervals[:-1]))
    # Truncate the sequence to the specified maximum duration
    data = np.cumsum(intervals)
    #state.order_popuptime = np.round(np.minimum(data, max_duration), decimals=0)
    state.order_popuptime = np.minimum(data, max_duration)
    # Accumulate the intervals to get the sequence of data
    ON = len(state.order_popuptime)

    ######## todo:   change a new order generator here
    # plan 1
    state.order_node = np.random.choice(np.linspace(0, NN - 1, NN), size = ON)

    # plan 2:
    """global predict_time_grain
    predcit_T = totaltime / predict_time_grain
    assert totaltime % predict_time_grain == 0, f'wrong in total time: {totaltime} and predict time grain:{predict_time_grain}.'
    poptime_predit_grain = get_random(min_v=0, max_v=predcit_T-1, n = ON) # 预测的时候用到
    state.order_popuptime = []
    for pred_t_idx in poptime_predit_grain:
        idx = np.random.choice(list(range(predict_time_grain)))
        popup_t = int(pred_t_idx) * predict_time_grain + idx
        state.order_popuptime.append(popup_t)
    state.order_popuptime = np.sort(state.order_popuptime)
    state.order_node = [int(_) for _ in get_random(min_v=0, max_v=NN - 1, n= ON)] #预测的时候用到"""
    state.order_timewindow= np.random.randint(low=TWlb, high=TWub, size=ON)
    global high_stay_bound, low_stay_bound
    if high_stay_bound == low_stay_bound: high_stay_bound += 1
    state.order_stay_time = np.random.randint(low=low_stay_bound, high=high_stay_bound, size=ON)
    # since at time 0 there may not be a delivery request, we add one artifically
    # add the location of the first order.
    state.postman_prev_node = np.random.choice(np.linspace(0, NN - 1, NN), size=PN)
    xcoords, ycoords = g.get_vertices()
    for p_id, node in enumerate(state.postman_prev_node):
        node = int(node)
        state.postman_xy.append([xcoords[node], ycoords[node]])
        g.set_postman_node(p_id, node)
    totalorders=len(state.order_popuptime)
    state.order_indice=np.linspace(0, totalorders - 1, totalorders)
    if randomseed!=None: np.random.seed()
    return g, state

def derive_current_location(G, path, prevedge, prevpercentage, elapsed, stayingtime_local, state, postmannum):
    Edges = G.get_edges()
    Coordsx, Coordsy = G.get_vertices()
    Nodesx = [Coordsx[p] for p in path]
    Nodesy = [Coordsy[p] for p in path]
    if path[0] == path[1]:
        Distance = [0]
    else:
        Distance = [Edges[path[p]][path[p + 1]] for p in range(len(path) - 1)]
    prevedge_index = path.index(prevedge[1])
    traverseddistance = sum(Distance[0:path.index(prevedge[0])])
    traverseddistance += prevpercentage * Distance[prevedge_index - 1]
    remaining = (1 - prevpercentage) * Distance[prevedge_index - 1]
    remaining1 = remaining

    def update_state_and_return(newlocation, newedge, traversedpercentage, nexttransit, current_node=None):
        state.exp_travel_time[postmannum] = nexttransit - stayingtime_local
        return newlocation, newedge, traversedpercentage, nexttransit, current_node

    def get_new_location(index, percentage):
        return [
            Nodesx[index] + percentage * (Nodesx[index + 1] - Nodesx[index]),
            Nodesy[index] + percentage * (Nodesy[index + 1] - Nodesy[index])
        ]

    if remaining >= elapsed * speed + 0.000001:
        traversedpercentage = (prevpercentage * Distance[prevedge_index - 1] + elapsed * speed) / Distance[
            prevedge_index - 1]
        newlocation = get_new_location(path.index(prevedge[0]), traversedpercentage)
        nexttransit = (sum(Distance) - traverseddistance) / speed - elapsed + stayingtime_local
        return update_state_and_return(newlocation, [prevedge[0], prevedge[1]], traversedpercentage, nexttransit)

    if remaining > elapsed * speed - 0.000001 and remaining < elapsed * speed + 0.000001:
        newlocation = [Coordsx[prevedge[1]], Coordsy[prevedge[1]]]
        nexttransit = (sum(Distance) - traverseddistance) / speed - elapsed + stayingtime_local

        if prevedge[1] == path[-1]:  # arrived at the destination
            if stayingtime_local == 0:  # arrive and finished
                state.already_stayed[postmannum] = 0
                state.exp_travel_time[postmannum] = 0  # will not travel any more
                return newlocation, [prevedge[1], prevedge[1]], 1, totaltime, path[-1]
            else:  # arrived but not finished delivering.
                if elapsed == 0:  # already stayed here before, since remaining is 0 accidentally recognised wrongly
                    # elapse should remain unchanged
                    return newlocation, [prevedge[1], prevedge[1]], 1, stayingtime_local - state.already_stayed[
                        postmannum], None
                else:
                    state.already_stayed[postmannum] = 0  # +=elapsed
                    state.postman_prev_node[postmannum] = path[-1]
                    state.exp_travel_time[postmannum] = - stayingtime_local
                    return newlocation, [prevedge[1], prevedge[1]], 1, stayingtime_local, None
        else:  # not arriving at the destination
            return update_state_and_return(newlocation, [path[prevedge_index], path[prevedge_index + 1]], 0,
                                           nexttransit)

    else:
        remaining = elapsed * speed - remaining1
        for d in range(len(Distance[prevedge_index:None])):
            if remaining < -0.000001 + Distance[prevedge_index + d] and remaining >= -0.000001:
                traversedpercentage = remaining / Distance[prevedge_index + d]
                newlocation = get_new_location(prevedge_index + d, traversedpercentage)
                nexttransit = stayingtime_local + (sum(Distance[prevedge_index + d:None]) - remaining) / speed
                state.already_stayed[postmannum] = 0
                return update_state_and_return(newlocation, [path[prevedge_index + d], path[prevedge_index + d + 1]],
                                               traversedpercentage, nexttransit)
            else:
                remaining -= Distance[prevedge_index + d]
        if remaining / speed <= stayingtime_local and remaining >= -0.00001:  # can travel to the destination, but before the staying time ends
            if int(state.postman_prev_node[postmannum]) == int(path[-1]):
                remainingstaytime = stayingtime_local - state.already_stayed[postmannum] - remaining / speed
            else:
                remainingstaytime = stayingtime_local - remaining / speed
            if remainingstaytime > 0.00001:  # has arrived not finished
                state.already_stayed[postmannum] += remaining / speed
                state.postman_prev_node[postmannum] = path[-1]
                state.exp_travel_time[postmannum] = -remaining / speed  # already arrived for how long
                return [Coordsx[path[-1]], Coordsy[path[-1]]], [path[-2], path[-1]], 1, remainingstaytime, None
            if remainingstaytime >= -0.00001 and remainingstaytime <= 0.00001:  # has arrived and finished
                state.already_stayed[postmannum] = 0
                state.exp_travel_time[postmannum] = -stayingtime_local
                return [Coordsx[path[-1]], Coordsy[path[-1]]], None, None, totaltime, path[-1]
        else:  # can travel to the destination and the stay time ends, finished
            state.already_stayed[postmannum] = 0
            state.exp_travel_time[postmannum] = -stayingtime_local
            return [Coordsx[path[-1]], Coordsy[path[-1]]], None, None, totaltime, path[-1]


def Environment(g, state, initial=False):
    prev_percentage= state.all_tra_percentage[:]
    prev_elapse=copy.copy(state.elapse)
    xcoord, ycoord = g.get_vertices()
    if state.elapse == totaltime and state.Done != True:
        if is_all_none(state.postman_destination_node) and len(state.order_node) == 0:
            state.Done = True
            return g, state
    #if displayoutlet:
    #remove the status of postman initial location, todo：这里的inital主要在做一件什么事情？
    if initial:
        locations_to_be_deleted=[]
        for i in range(len(state.order_node)):
            if state.order_popuptime[i] == 0:
                locations_to_be_deleted.append(i)
            else:
                break
        #----------------------------------------------------
        # todo: 修改数据结构之后的
        state.order_node = np.delete(state.order_node, locations_to_be_deleted)
        state.order_popuptime = np.delete(state.order_popuptime, locations_to_be_deleted)
        state.order_stay_time = np.delete(state.order_stay_time, locations_to_be_deleted)
        state.order_indice=np.delete(state.order_indice, locations_to_be_deleted)
        state.order_timewindow=np.delete(state.order_timewindow, locations_to_be_deleted)
        #----------------------------------------------------

        state.postman_current_path = []
        state.all_edges = []
        Distance = []
        assert len(state.postman_destination_node) == len(state.postman_prev_node), "Error,wrong destination size"

        for i in range(len(state.postman_destination_node)):
            if state.postman_prev_node[i]==state.postman_destination_node[i]:
                g.update_postman(xcoord[int(state.postman_prev_node[i])], ycoord[int(state.postman_prev_node[i])], i, 1, 1) # still working, has arrived
                pin = int(state.postman_prev_node[i])
                state.postman_current_path.append([pin, pin])
                state.all_edges.append([pin, pin])
            else:
                if state.postman_destination_node[i]!=None:
                    g.update_postman(xcoord[int(state.postman_prev_node[i])], ycoord[int(state.postman_prev_node[i])], i, 1, 0)#assigned, not arrived
                else:
                    g.update_postman(xcoord[int(state.postman_prev_node[i])], ycoord[int(state.postman_prev_node[i])], i, 0, 0) #unassigned, not arrived
                if state.postman_destination_node[i] != None:
                    D, path = g.find_distance(state.postman_prev_node[i], state.postman_destination_node[i])
                    state.postman_current_path.append(path)
                    Distance.append(D / speed + state.postman_stayingtime[i])
                    state.all_edges.append([path[0], path[1]])
                    state.exp_travel_time[i] = D / speed
                else:
                    state.all_edges.append(None)
                    state.postman_current_path.append(None)
        state.all_tra_percentage = np.zeros(len(state.postman_destination_node))
        prev_percentage=[0 for i in range(state.PN)]
    else:
        # 每个快递员之前走在那条边上，
        assert prev_elapse >= 0 and prev_elapse <= state.current_time, f'error, negative forward time, prev_elapse:{prev_elapse}'
        locations_to_be_deleted = []
        for i in range(len(state.order_popuptime)):
            if state.current_time - state.elapse >= state.order_popuptime[i] - 0.0001: #state.popuptime是一个沿着时间严格递增的序列
                 # if this customer has been added to the graph then we can remove it
                if state.order_popuptime[i] != 0:
                    #g.add_order_node(state.order_node[i])
                    locations_to_be_deleted.append(int(i))
                    # memorise the indices, delete later. to avoid wrong indices
            else:
                break
        state.order_node = np.delete(state.order_node, locations_to_be_deleted)
        state.order_popuptime = np.delete(state.order_popuptime, locations_to_be_deleted)
        state.order_stay_time=np.delete(state.order_stay_time, locations_to_be_deleted)
        state.order_timewindow = np.delete(state.order_timewindow, locations_to_be_deleted)
        state.order_indice=np.delete(state.order_indice, locations_to_be_deleted)
        finished_to_remove = []
        for i in range(len(state.delivered_node)):
            if state.postman_destination_node[state.delivered_postman[i]] != None:
                # note this can be problematic in later stages, because it removes the first customer in this list at this location
                # this may not exactly be the correct location
                finished_to_remove.append(i) # i here is the index in finishedtas, not the task index
                D, state.postman_current_path[state.delivered_postman[i]] =\
                    g.find_distance(state.postman_prev_node[state.delivered_postman[i]],
                                    state.postman_destination_node[state.delivered_postman[i]])
                state.all_edges[state.delivered_postman[i]] = \
                    [state.postman_current_path[state.delivered_postman[i]][0],
                     state.postman_current_path[state.delivered_postman[i]][1]]
                prev_percentage[state.delivered_postman[i]] = 0
            else:
                pass
        state.delivered_node = [state.delivered_node[i] for i in range(len(state.delivered_node)) if i not in finished_to_remove]
        state.delivered_postman = [state.delivered_postman[i] for i in range(len(state.delivered_postman)) if i not in finished_to_remove]
    prevedge = state.all_edges[:]
    allnextelapses = []
    state.all_tra_percentage = []
    state.all_edges = []
    for i in range(state.PN):
            if state.postman_destination_node[i] == None:  # if this postman is not assigned a new mission
                allnextelapses.append(totaltime)
                state.all_tra_percentage.append(0)
                state.all_edges.append(None)
                state.exp_travel_time[i]=0
                if displayoutlet:
                    print('postman',i,'exit1')
            else:
                if prevedge[i] == None:
                    # if there is a destination but no edge information, must be from a previous elapse step, then add an edge information
                    dummy, state.postman_current_path[i] =g.find_distance(state.postman_prev_node[i],
                                                                          state.postman_destination_node[i])
                    prevedge[i] = [int(state.postman_current_path[i][0]), int(state.postman_current_path[i][1])]
                    prev_percentage[i] = 0
                    #if the destination is accidentally the start node
                    if state.postman_stayingtime[i] == 0 and state.postman_current_path[i][1] == state.postman_current_path[i][0]: #set the prevpercentage to 1, then later it will be processed
                        prev_percentage[i] = 1
                if prevedge[i][0] != prevedge[i][1]:  # if the destination is NOT accidentally at the start node
                    newPN, edgenode, traversedpercentage, remainingelapsetime, delivered = derive_current_location(
                        g,
                        state.postman_current_path[i],
                        prevedge[i],
                        prev_percentage[i],
                        prev_elapse,
                        state.postman_stayingtime[i],state,i)
                    allnextelapses.append(remainingelapsetime)
                    if delivered != None:  # delivered is the destination of delivery
                        g.add_delivered(i,
                                        delivered,
                                        state.current_time,
                                        state.postman_destination_indice[i])
                        if displayoutlet:
                            print('postman', i, 'finishes task index', state.postman_destination_indice[i], 'at',
                              state.current_time,'dest',delivered)
                        #i is postman, delivered is the destination of delivery,
                        state.postman_prev_node[i] = delivered
                        g.erase_order_node(delivered)  # erase this task if it is finished by its location.
                        state.delivered_postman.append(i)
                        state.delivered_node.append(delivered)
                        g.set_postman_node(i, delivered)
                        state.postman_destination_node[i] = None
                        state.postman_destination_indice[i]=None
                        g.update_postman(newPN[0], newPN[1], i, 0, 1) #already delivered.
                    else:
                        g.update_postman(newPN[0], newPN[1], i, 1, traversedpercentage) #still working not delivered
                    state.all_tra_percentage.append(traversedpercentage)
                    state.all_edges.append(edgenode)
                else:  #same destination and initial node
                    state.all_tra_percentage.append(1)
                    state.all_edges.append(None)
                    #problematic. should not be popuptime must be the actualassignmenttime.
                    passedstayingtime = -state.postman_assignment_time[i] + state.current_time - state.postman_current_path_distance[i] / speed
                    if passedstayingtime >= state.postman_stayingtime[i]-0.00001:  # already arrived, already finished delivery
                        state.postman_prev_node[i] = g.get_postman_node(i)
                        state.delivered_node.append(int(state.postman_destination_node[i]))
                        g.erase_order_node(int(state.postman_destination_node[i]))  # erase this task if it is finished by its location.
                        g.add_delivered(i,
                                        int(state.postman_destination_node[i]),
                                        state.current_time,
                                        state.postman_destination_indice[i])
                        g.set_postman_node(i, int(state.postman_destination_node[i]))
                        state.delivered_postman.append(i)
                        state.already_stayed[i]=0
                        state.postman_destination_node[i] = None
                        state.postman_destination_indice[i]=None
                        allnextelapses.append(totaltime)
                        g.update_postman(xcoord[prevedge[i][0]], ycoord[prevedge[i][0]], i, 0, 1) #idling
                    else:#arrived but not finished delivery
                        allnextelapses.append(state.postman_stayingtime[i] - passedstayingtime)
                        g.update_postman(xcoord[prevedge[i][0]], ycoord[prevedge[i][0]], i, 1, 1)  # still wroking
                    state.exp_travel_time[i]=0
        
    nextelapse = min(allnextelapses)
    state.elapse=nextelapse
    for i in range(state.PN):
        if state.postman_prev_node[i] != state.postman_destination_node[i] and state.all_tra_percentage[i]!=0 and g.postman_status==0:
            state.postman_prev_node[i] = None
    if len(state.order_node)==0 and len(g.get_current_order_node())==0 and is_all_none(state.postman_destination_node):
        state.Done=True

    return g, state

# pop up time: at every interval popup customers that have been added should be removed
# location,state.stayingtime should be removed as well
# below is a random generator, generate random policies
# ----------------------------
def greedy_policy(state: State, g: Graph, PN, initial=False):
    if displayoutlet:
        print('state.current_time: ', state.current_time)
        print('state.elapse: ',state.elapse)
    if not initial:
        for pt in state.order_popuptime:
            delta_t = pt - state.current_time
            if delta_t > 0:
                state.elapse = min(state.elapse, delta_t)
                break
        if state.elapse==totaltime:
            if len(state.unassigned_order_node) == 0 and len(state.order_node) == 0:  # all assignments are made
                # --------------------do not delete------------------------------
                #   state.Done=True very suspicious
                state.current_time = state.current_time + state.elapse
                return g, state
            # -------------------------------------------------------
            elif len(state.unassigned_order_node) != 0:  # there are still remaining tasks at the same location the postman stays
                state.elapse = min(state.unassigned_order_staytime)
        # need to add another thing here, stating that if further elapse, then the first one in the list should be assigned a postman
        # then evaluate the shortest travel duration, then set the elapse time.
        state.change_elapse=[False for _ in range(len(state.unassigned_order_node))]
        for i in range(len(state.order_node)):
            if state.current_time >= state.order_popuptime[i] > state.previous_time:
                state.unassigned_order_indice.append(int(state.order_indice[i]))
                state.unassigned_order_node.append(int(state.order_node[i]))
                state.unassigned_order_popuptime.append(state.order_popuptime[i])
                state.unassigned_order_staytime.append(state.order_stay_time[i])
                state.unassigned_order_timewindow.append(state.order_timewindow[i])
                state.change_elapse.append(True)
                g.add_order_node(state.order_node[i])
        assert len(state.order_popuptime)==len(state.order_node), "wrong size"
    else:
        for i in range(len(state.order_node)):
            if state.order_popuptime[i] <= 0:
                state.unassigned_order_indice.append(int(state.order_indice[i]))
                state.unassigned_order_node.append(int(state.order_node[i]))
                state.unassigned_order_popuptime.append(state.order_popuptime[i])
                state.unassigned_order_staytime.append(state.order_stay_time[i])
                state.unassigned_order_timewindow.append(state.order_timewindow[i])
                g.add_order_node(state.order_node[i])
        state.change_elapse = [False for _ in range(len(state.unassigned_order_node))]
    assert len(state.unassigned_order_node)==len(state.unassigned_order_timewindow), 'wrong different sizes of timewindow and orderlist'
    state.postman_elapse = [None for _ in range(PN)]
    state.previous_time  = state.current_time
    if state.unassigned_order_node == []:  # no new customer in this time interval
        state.postman_elapse = [None for _ in range(PN)]
        state.current_time = state.current_time + state.elapse
        return g, state
    if state.postman_destination_node == False or state.postman_destination_node == [] or state.postman_destination_node == None:
        state.postman_destination_node = [None for _ in range(PN)]
        state.postman_destination_indice=[None for _ in range(PN)]
    # combination of delivered and State.postmandestinations
    idling_postmen = [p_idx for p_idx, des_node in enumerate(state.postman_destination_node) if des_node==None]
    OD=np.zeros([len(idling_postmen), len(state.unassigned_order_node)])
    all_pathes = {}
    for i, node in enumerate(state.unassigned_order_node):
        for j, postman in enumerate(idling_postmen):
            distance, path = g.find_distance(start=state.postman_prev_node[int(postman)], end=node)
            OD[j, i] = distance # evaluate the distance from this postman to every destination this is euclidean distance
            all_pathes[j, i] = path
    assignment, selected_postmen = Greedyoptimiser(OD, idling_postmen, len(state.unassigned_order_node))
    inner_counter=0 #reflects the index of postman in the OD matrix
    delete_order=[]
    for i in assignment.keys(): # value corresponds to OD location index, key corresponds to postman
        assigned_index = assignment[i] # OD index also the index of unassignedorder
        current_postman = selected_postmen[inner_counter]
        state.postman_destination_node[int(i)] = int(state.unassigned_order_node[assigned_index]) #assign destination
        state.postman_destination_indice[int(i)] = int(state.unassigned_order_indice[assigned_index])
        assignment_time = max(state.unassigned_order_popuptime[assigned_index], state.current_time)#-state.elapse)
        g.add_assignment(int(i),
                         state.unassigned_order_node[assigned_index],
                         assignment_time,
                         OD[current_postman, assigned_index],
                         state.unassigned_order_popuptime[assigned_index],
                         state.unassigned_order_indice[assigned_index],
                         state.unassigned_order_timewindow[assigned_index], 'greedy',
                         state.unassigned_order_staytime[assigned_index])
        state.postman_current_path_distance[i] = OD[current_postman, assigned_index]
        if displayoutlet:
            print('greedy benchmark assign postman', int(i),'with loc', state.unassigned_order_node[assigned_index], 'index', state.unassigned_order_indice[assigned_index],
              'distance', OD[current_postman,assigned_index],'popuptime', state.unassigned_order_popuptime[assigned_index], 'at', assignment_time, 'will stay',
                  state.unassigned_order_staytime[assigned_index])

        distance1, path = g.find_distance(start=state.postman_prev_node[i], end=state.unassigned_order_node[assigned_index])
        assert distance1==OD[current_postman, assigned_index], 'error'

        state.postman_stayingtime[i] = state.unassigned_order_staytime[assigned_index]
        state.postman_assignment_time[i] = assignment_time
        #state.postmanelapse[int(i)]=state.currenttime-assignmenttime
        state.elapse = min(state.elapse, distance1 / speed + state.unassigned_order_staytime[assigned_index])
        delete_order.append(assigned_index)
        inner_counter+=1

    state.change_elapse = delete_by_idx(state.change_elapse, delete_order)
    state.unassigned_order_popuptime = delete_by_idx(state.unassigned_order_popuptime, delete_order)
    state.unassigned_order_timewindow = delete_by_idx(state.unassigned_order_timewindow, delete_order)
    state.unassigned_order_node = delete_by_idx(state.unassigned_order_node, delete_order)
    state.unassigned_order_staytime = delete_by_idx(state.unassigned_order_staytime, delete_order)
    state.unassigned_order_indice = delete_by_idx(state.unassigned_order_indice, delete_order)
    state.current_time = state.current_time + state.elapse
    if state.elapse == totaltime:
        if state.unassigned_order_node == [] and state.postman_destination_node == [None for i in range(state.PN)]:
            state.Done = True
            print('greedy optimiser elapse', state.elapse)
            print('exit3')
            return g, state
        else:
            print('error, greedy policy wrong, elapse=totaltime')
            raise KeyboardInterrupt
    return g, state

def episode_reward(popup_times, postman_assigned_index, delivered_times, postman_delivered_index, time_windows, staying_times,assigned_distance):
    for i in range(len(popup_times)):
        if list(np.sort(postman_assigned_index[i]))!=list(np.sort(postman_delivered_index[i])):
            print('error inconsistent assignment and delivered orders')
            print(np.sort(postman_assigned_index[i]))
            print(np.sort(postman_delivered_index[i]))
            sys.exit()
    penalty=sum([max(delivered_times[i][j] - (popup_times[i][j] + time_windows[i][j] + staying_times[i][j]), 0)
                 for i in range(len(popup_times)) for j in range(len(popup_times[i]))])
    variance=np.var([max(delivered_times[i][j] - (popup_times[i][j] + time_windows[i][j] + staying_times[i][j]),0)
                 for i in range(len(popup_times)) for j in range(len(popup_times[i]))])
    averagelateness=np.mean([max(delivered_times[i][j] - (popup_times[i][j] + time_windows[i][j] + staying_times[i][j]), 0)
                 for i in range(len(popup_times)) for j in range(len(popup_times[i]))])
    #if too early no penalty
    penalty2=sum([sum(assigned_distance[i]) for i in range(len(assigned_distance))])
    #warning this reward function is not strictly true. It is for safety concerns that each i and j are related.
    return penalty,float(penalty2), averagelateness, variance