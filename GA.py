import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import sys
import torch
import numpy as np
import Parameters
import Environment as Env
from more_itertools import sort_together
import pandas as pd
import time
from geneticalgorithm import geneticalgorithm as ga
from python_tsp.heuristics import solve_tsp_simulated_annealing
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from more_itertools import sort_together
if_output_solutions = False
display_details = False
args = Parameters.parameters()
speed = vars(args)['speed']
totaltime = vars(args)['totaltime']
lb = vars(args)['low_order_bound']
hb = vars(args)['high_order_bound']
gap = 4
global distance
global PN, NN
PN = vars(args)['PNlower']
global ODmatrix
from utils.my_utils import dir_check, flatten, write_xlsx, multi_thread_work, Exp_Buffer, is_all_none
local_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
path = f'./output/{local_time}/'
dir_check(path)
def flatten(t):
    return [item for sublist in t for item in sublist]



def derivepath(sequence):
    indices = {}
    # UNtoAllN: map Unassigned to all nodes
    path = []
    totalpenalty = 0
   # print(sequence)
    for i in range(PN):
        indices[i] = [index for index in range(len(sequence)) if sequence[index] == i]
        alllocations = indices[i]
        if len(alllocations) != 0:
            OD = np.zeros((len(alllocations) + 1, len(alllocations) + 1))
            for k1 in range(1, len(alllocations) + 1):
                for k2 in range(1, len(alllocations) + 1):
                    if k1 != k2:
                        OD[k1, k2] = ODmatrix[UNtoAllN[alllocations[k1 - 1]], UNtoAllN[alllocations[k2 - 1]]]
                    else:
                        OD[k1, k2] = 0
            for k in range(1, len(alllocations) + 1):
                OD[0, k] = ODmatrix[postmancurrentlocations[i], UNtoAllN[alllocations[k - 1]]]

            if len(alllocations) == 1:
                distance = OD[0, 1]
                path.append([UNtoAllN[alllocations[0]]])
            else:
                permutation, distance = solve_tsp_simulated_annealing(OD)
                path.append([UNtoAllN[
                                 alllocations[j-1]]
                             for j in permutation[1:None]])
            totalpenalty += distance
        else:
            path.append([])
    return path, totalpenalty

def findeucdistance(x,y):
    return ((x[0]-y[0])**2+(x[1]-y[1])**2)**0.5

def environment(sequence):
    indices={}
    #UNtoAllN: map Unassigned to all nodes
    path={}
    totalpenalty=0
   # print(sequence)
   # print('OD',ODmatrix)
   # print('untoall',UNtoAllN)

    for i in range(PN):
        indices[i]=[index for index in range(len(sequence)) if sequence[index]==i]
        alllocations=indices[i]
      #  print('allloc',alllocations)
        if len(alllocations)!=0:
            OD=np.zeros((len(alllocations)+1,len(alllocations)+1))
            for k1 in range(1, len(alllocations)+1):
                for k2 in range(1, len(alllocations)+1):
                    if k1!=k2:
                       # print('untoall',UNtoAllN[alllocations[k1-1]])
                        OD[k1,k2]=ODmatrix[UNtoAllN[alllocations[k1-1]], UNtoAllN[alllocations[k2-1]]]
                    else:
                        OD[k1,k2]=0
            for k in range(1, len(alllocations)+1):
              #  print(postmancurrentlocations[i])
                OD[0, k]=ODmatrix[postmancurrentlocations[i], UNtoAllN[alllocations[k-1]]]
            if len(alllocations)==1:
                distance=OD[0,1]
                permutation=[alllocations[0]]
            else:
                start=time.time()
                permutation, distance = solve_tsp_simulated_annealing(OD)
                end=time.time()-start
              #  print('caltime', end)
           # path[i] = [alllocations[j] for j in permutation[1:None]
            totalpenalty+=distance
        else:
            totalpenalty+=10
    return totalpenalty

def optimise(unassignedorders):
    NN=len(unassignedorders)
    varbound=np.array([[0,PN-1]]*NN)
   # sequence=[1 for i in range(NN)]
  #  print(UNtoAllN)
  #  print(allnodeslocations)
    model=ga(function=environment,dimension=NN,variable_type='int',variable_boundaries=varbound, convergence_curve=False)
    model.run()
    solution=model.output_dict
  #  print(list(solution['variable']))
    path, totalpenalty = derivepath(list(solution['variable']))
  #  print('path', path,'penalty', totalpenalty)
    return path
# static optimiser, can be used for recursive reoptimisation


# ------------------------Testing Module-----------------
def RecursiveReoptimisation(PN, NN, graphseed, ep,path):

    global UNtoAllN, ODmatrix, postmancurrentlocations, allnodeslocations
    starttime = time.time()
    device = torch.device('cpu')
    g1 = Env.Graph(PN)  # initialise a graph
    g1 = Env.graph_generator(g1, NN, randomseed=graphseed, graphname=None, Mag_Factor=1)
    state = Env.State(PN, NN, device)
    g1, state= Env.random_generator(g1, lb, hb, NN, PN, state, randomseed=ep)
   # print(state.order_popuptime)
    state.Done = False
    averagedeliverytime = np.mean(state.order_stay_time)
    prevedge = [[state.postman_prev_node[i], state.postman_prev_node[i]] for i in range(PN)]
    postmen = np.linspace(0, PN - 1, PN)  # for initial state all postmen are unselected
    # unassignedorders=state.unassignedorder
    # timewindows=[state.unassignedtimewindow[i]+state.unassignedpopuptime[i] for i in range(len(state.unassignedtimewindow))]
    # deliverytime=state.unassignedstayingtime
    for i in range(PN):
        g1.postman_percentage[i] = 0
    postmeninitiallocation = [prevedge[i][1] for i in range(PN)]
    # ------------------------------------preparation steps------------------------
    state.postman_destination_node = [None for i in range(PN)]
    postmantrajectory = [[] for i in range(PN)]
    trajectorytimewindows = [[] for i in range(PN)]
    currentgap = 0
    counter = 0
    state.elapse = 0  # no elapse at the initial time step]
    state.postman_destination_indice = [None for _ in range(PN)]
    alldelivered = 0
    if display_details:
        print('totalorders', len(state.order_node))
    # ------------------------------------start iteration ------------------------

    while state.Done != True:
        state.postman_elapse = [None for i in range(PN)]
        # if state.elapse == totaltime:
        #     if len(state.unassignedorder) == 0 and postmantrajectory==[[] for _ in range(PN)]:  # all assignments are made
        #          print('criteriareached')
        #          state.Done = True

        print('elapse',state.elapse)
        print('currentgap',currentgap)


        if state.current_time >= currentgap:  # deadline triggered, need a re-optimisation, this should never be the initial case
            # print('enter reoptimisation')
            gapelapse = currentgap - state.current_time
            # print(currentgap,state.currenttime,state.elapse)
            currentgap += gap
            if counter == 0:  # if initial, no need to elapse
               # state.elapse = 0
                for i in range(len(state.order_node)):
                    if state.order_popuptime[i] == 0:
                        state.unassigned_order_node = state.unassigned_order_node + [int(state.order_node[i])]
                        state.unassigned_order_indice = state.unassigned_order_indice + [int(state.order_indice[i])]
                        state.unassigned_order_popuptime = state.unassigned_order_popuptime + [state.order_popuptime[i]]
                        state.unassigned_order_staytime = state.unassigned_order_staytime + [state.order_stay_time[i]]
                        state.unassigned_order_timewindow = state.unassigned_order_timewindow + [
                            state.order_timewindow[i]]
                        g1.add_order_node(state.order_node[i])
            else:
              #  state.elapse = gapelapse
              #  state.current_time = gapelapse + state.current_time
                for i in range(len(state.order_node)):
                    if state.current_time >= state.order_popuptime[i] > state.previous_time:
                        state.unassigned_order_node = state.unassigned_order_node + [int(state.order_node[i])]
                        state.unassigned_order_indice = state.unassigned_order_indice + [int(state.order_indice[i])]
                        state.unassigned_order_popuptime = state.unassigned_order_popuptime + [state.order_popuptime[i]]
                        state.unassigned_order_staytime = state.unassigned_order_staytime + [state.order_stay_time[i]]
                        state.unassigned_order_timewindow = state.unassigned_order_timewindow + [
                            state.order_timewindow[i]]
                        g1.add_order_node(state.order_node[i])
            print('unassigned', state.unassigned_order_node)
            for i in range(PN):
                if state.postman_destination_node[i] != None:
                    postmeninitiallocation[i] = state.postman_destination_node[i]
                else:  # the postman may remain stationary somewhere, therefore we take the postman initial node
                    postmeninitiallocation[i] = state.postman_prev_node[i]
            if display_details:
                print(state.unassigned_order_node)
            if state.unassigned_order_node == []:  # at gap time, no new order emerges
                pass
            else:
                unassignedorders = state.unassigned_order_node
                timewindows = [state.unassigned_order_timewindow[i] + state.unassigned_order_popuptime[i] for i in
                               range(len(state.unassigned_order_timewindow))]
               # postmantrajectory, postmanarrivaltimes, trajectorytimewindows \
               #     = optimise(g1, postmeninitiallocation, postmen, unassignedorders, timewindows, averagedeliverytime)
                UNtoAllN = unassignedorders
                ODmatrix = g1.ODlist
                postmancurrentlocations=[]
                for p in range(PN):
                    if state.postman_destination_node[p]!=None:
                        postmancurrentlocations.append(int(state.postman_destination_node[p]))
                    else:
                        postmancurrentlocations.append(int(state.postman_prev_node[p]))
                allnodeslocations = [[g1.xcoord[k], g1.ycoord[k]] for k in range(NN)]
                postmantrajectory = optimise(unassignedorders)
                if len(timewindows) != len(flatten(postmantrajectory)) or len(timewindows) != len(unassignedorders):
                    print(len(timewindows))
                    print(len(flatten(postmantrajectory)))
                    print('error wrong output size')
                    sys.exit()
                allnextelapses=[]
                print('traj after assignment',postmantrajectory)
                for i in range(PN):
                    if postmantrajectory[i] != [] and state.postman_destination_node[
                        i] == None:  # only trigger if postman destination is arrived
                        assignedorder = postmantrajectory[i][0]
                        state.postman_destination_node[i] = assignedorder
                        del postmantrajectory[i][0]
                     #   del trajectorytimewindows[i][0]
                        assignedindex = state.unassigned_order_node.index(assignedorder)
                        state.postman_destination_indice[i] = state.unassigned_order_indice[assignedindex]
                        distance1, _ = g1.find_distance(start=state.postman_prev_node[i],
                                                        end=state.postman_destination_node[i])
                        print('assign',i,'from',state.postman_prev_node[i], 'to',int(state.unassigned_order_node[assignedindex]),
                              'dist',distance1/speed,'stay', state.unassigned_order_staytime[assignedindex])
                        g1.add_assignment(i, int(state.unassigned_order_node[assignedindex]),
                                          state.current_time, distance1,
                                          state.unassigned_order_popuptime[assignedindex],
                                          state.unassigned_order_indice[assignedindex],
                                          state.unassigned_order_timewindow[assignedindex], 'VRPTWOptimiser',
                                          state.unassigned_order_staytime[assignedindex])
                        state.postman_stayingtime[i] = state.unassigned_order_staytime[assignedindex]
                        state.postman_assignment_time[i] = state.current_time
                        state.postman_current_path_distance[i] = distance1
                        allnextelapses.append(distance1 / speed + state.unassigned_order_staytime[assignedindex])
                        state.postman_elapse[i] = 0
                        if display_details:
                            print('assign postman', i, 'to', state.unassigned_order_node[assignedindex], 'at',
                                  state.current_time, 'distance', distance1)
                        alldelivered += 1
                        del state.unassigned_order_node[assignedindex]
                        del state.unassigned_order_indice[assignedindex]
                        del state.unassigned_order_popuptime[assignedindex]
                        del state.unassigned_order_staytime[assignedindex]
                        del state.unassigned_order_timewindow[assignedindex]

          #  state.elapse=min(allnextelapses)
        else:
            # state.currenttime += state.elapse
            #  print('continuation')
            allnextelapses = []
            for i in range(len(state.order_node)):
                if state.current_time >= state.order_popuptime[i] > state.previous_time:
                    state.unassigned_order_node = state.unassigned_order_node + [int(state.order_node[i])]
                    state.unassigned_order_indice = state.unassigned_order_indice + [int(state.order_indice[i])]
                    state.unassigned_order_popuptime = state.unassigned_order_popuptime + [state.order_popuptime[i]]
                    state.unassigned_order_staytime = state.unassigned_order_staytime + [state.order_stay_time[i]]
                    state.unassigned_order_timewindow = state.unassigned_order_timewindow + [state.order_timewindow[i]]
                    g1.add_order_node(int(state.order_node[i]))
            print('unassigned', state.unassigned_order_node)
            for i in range(PN):
                if postmantrajectory[i] != [] and state.postman_destination_node[i] == None:
                    # only trigger if postman destination is arrived
                    assignedorder = postmantrajectory[i][0]
                    state.postman_destination_node[i] = postmantrajectory[i][0]
                    assignedindex = state.unassigned_order_node.index(assignedorder)
                    del postmantrajectory[i][0]
                 #   del trajectorytimewindows[i][0]
                    state.postman_destination_indice[i] = state.unassigned_order_indice[assignedindex]
                    distance1, _ = g1.find_distance(start=state.postman_prev_node[i],
                                                    end=state.postman_destination_node[i])
                    print('assign', i, 'from', state.postman_prev_node[i], 'to',
                          int(state.unassigned_order_node[assignedindex]),
                          'dist', distance1 / speed,'stay', state.unassigned_order_staytime[assignedindex])
                    g1.add_assignment(i, int(state.unassigned_order_node[assignedindex]),
                                      state.current_time, distance1, state.unassigned_order_popuptime[assignedindex],
                                      state.unassigned_order_indice[assignedindex],
                                      state.unassigned_order_timewindow[assignedindex], 'VRPTWOptimiser',
                                      state.unassigned_order_staytime[assignedindex])

                    state.postman_stayingtime[i] = state.unassigned_order_staytime[assignedindex]
                    state.postman_assignment_time[i] = state.current_time
                    state.postman_current_path_distance[i] = distance1
                   # state.elapse = min(state.elapse, distance1 / speed + state.unassigned_order_staytime[assignedindex])
                    allnextelapses.append(distance1 / speed + state.unassigned_order_staytime[assignedindex])
                    state.postman_elapse[i] = 0
                    # -----------------------statistics--------------------
                    del state.unassigned_order_node[assignedindex]
                    if display_details:
                        print('assign postman', i, 'to', state.unassigned_order_node[assignedindex], 'at',
                              state.current_time,
                              'distance', distance1)
                    alldelivered += 1
                    del state.unassigned_order_indice[assignedindex]
                    del state.unassigned_order_popuptime[assignedindex]
                    del state.unassigned_order_staytime[assignedindex]
                    del state.unassigned_order_timewindow[assignedindex]
                # if after optimisation, still one or more postmen have no destination assigned, then we can ignore (insufficient orders)
       # print('elapse1', state.elapse)
        print('postman', state.postman_destination_node)
        state.previous_time = state.current_time
        if counter == 0:
            state.elapse=0
            g1, state = Env.Environment(g1, state, initial=True)
        if len(allnextelapses)!=0: elapse1=min(allnextelapses)
        state.elapse = min(state.elapse, elapse1, currentgap - state.current_time)
        state.current_time = state.current_time + state.elapse
        print('---------currenttime', state.current_time)
        g1, state = Env.Environment(g1, state, initial=False)
        counter += 1
        print('postmantraj', postmantrajectory)


    postmanassignedtasks, postmanassignedtime, postmanassigneddistance, \
    customerpopuptime, postmanassignedindex, alltimewindows, assignmentmethod,stayingtimes,_ = g1.get_assigned()
    postmandeliveredtasks, postmandeliveredtimes, postmanorderindex = g1.get_delivered()
    print(postmanassignedindex)
    print(postmanassignedtasks)
    for i in range(PN):
        postmanassignedtasks[i] = sort_together([postmanassignedindex[i], postmanassignedtasks[i]])[1]
        postmanassignedtime[i] = sort_together([postmanassignedindex[i], postmanassignedtime[i]])[1]
        customerpopuptime[i] = sort_together([postmanassignedindex[i], customerpopuptime[i]])[1]
        postmanassigneddistance[i] = sort_together([postmanassignedindex[i], postmanassigneddistance[i]])[1]
        postmanassignedindex[i] = sort_together([postmanassignedindex[i], postmanassignedindex[i]])[1]
        assignmentmethod[i] = sort_together([postmanassignedindex[i], assignmentmethod[i]])[1]
        postmandeliveredtimes[i] = sort_together([postmanassignedindex[i], postmandeliveredtimes[i]])[1]
        postmanorderindex[i] = sort_together([postmanassignedindex[i], postmanorderindex[i]])[1]

    episodereward = Env.Cal_episode_reward(customerpopuptime,
                                       postmanassignedindex,
                                       postmandeliveredtimes,
                                       postmanorderindex,
                                       alltimewindows,
                                       stayingtimes,
                                       postmanassigneddistance)
    actionlength = len(flatten(postmanassignedtasks))
    evaluationtime = [time.time() - starttime] + [0 for _ in range(actionlength - 1)]
    episodereward1 = [episodereward] + [0 for _ in range(actionlength - 1)]
    dfdict = {'assigned tasks': flatten(postmanassignedtasks),
              'assigned time': flatten(postmanassignedtime),
              'request time': flatten(customerpopuptime),
              'timewindows': flatten(alltimewindows),
              'assigned distance': flatten(postmanassigneddistance),
              'assignedindex': flatten(postmanassignedindex),
              'assignmentmethod': flatten(assignmentmethod),
              'delivered times': flatten(postmandeliveredtimes),
              'deliveredindex': flatten(postmanorderindex),
              'episodereward': episodereward1,
              'evaluationtime': evaluationtime}
    df = pd.DataFrame(data=dfdict)
    serverversion = False

    log_cfg = {'fout': path + 'GreedyStatistics.xlsx', "sheet_name": f'graph_{graphseed}',
               'server_version': serverversion, 'time': time.time() - starttime}
    episode_reward, _, averagelateness, variance, _, Avg_Idle_Time1,Avg_Travel_Dist1 = Env.calculate_and_log_reward(g1, log_cfg, PN)

    return df, episode_reward, averagelateness, variance, time.time()-starttime, sum(state.order_stay_time), Avg_Idle_Time1,Avg_Travel_Dist1

graphseed=0
num_episodes=100
eps=[i+1000 for i in range(num_episodes)]
alllateness=[]
allvar=[]
Caltime=[]
allstaytime=[]
Avg_Travel_Dist=[]
Avg_Idle_Time=[]
allrewards=[]
episodereward=[]
externalcounter=0

for ep in eps:
    df, episodereward, averagelateness, variance, caltime, staytime, Avg_Idle_Time1,Avg_Travel_Dist1 = RecursiveReoptimisation(PN, 8, graphseed, ep,path)
    allrewards.append(episodereward)
    alllateness.append(averagelateness)
    allvar.append(variance)
    Caltime.append(caltime)
    allstaytime.append(staytime)
    Avg_Travel_Dist.append(Avg_Travel_Dist1)
    Avg_Idle_Time.append(Avg_Idle_Time1)
    if externalcounter%5==0:
        test_results = pd.DataFrame({'reward': episodereward, 'lateness': alllateness,
                                     'var': allvar, 'caltime': Caltime,
                                 'allstaytime': allstaytime, 'Avg_Traveldist': Avg_Travel_Dist,
                                 'Avg_Idle_Time': Avg_Idle_Time})
        test_results.to_csv(path + 'GA_test_results.csv')
    externalcounter+=1

test_results = pd.DataFrame({'reward': episodereward, 'lateness': alllateness,
                             'var': allvar, 'caltime': Caltime,
                             'allstaytime': allstaytime, 'Avg_Traveldist': Avg_Travel_Dist,
                             'Avg_Idle_Time': Avg_Idle_Time})
test_results.to_csv(path + 'GA_test_results.csv')







