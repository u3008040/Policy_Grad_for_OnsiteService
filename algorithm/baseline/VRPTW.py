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
if_output_solutions=False
display_details=False
args = Parameters.parameters()
speed = vars(args)['speed']
totaltime = vars(args)['totaltime']
lb = vars(args)['lowerbound']
hb = vars(args)['higherbound']
gap=vars(args)['gap']
def flatten(t):
    return [item for sublist in t for item in sublist]

#static optimiser, can be used for recursive reoptimisation
def VRTPW(g,postmeninitiallocation,postmen,unassignedorders,timewindows,deliverytime):
    #find the required OD
    #postmeninitiallocation should be the initial node of each postman
    #postmen includes indices of postmen
    #unassignedorder includes indices of locations, may have multiple locations
    #timewindows should have the time windows of each unassigned location, may have multiple locations.
    #timewindows should have the absolute time windows not relative to the popup time
    PN=len(postmen)
    NN=len(unassignedorders)
    ODPO=np.zeros([PN,NN])
    M=totaltime #check if this is sufficient.
    x1={} #Xijk if postman p goes from destination i to j
    x2={} #Xijk if postman p goes from its original location to destination i as the first node to visit
    AT={} #AT is the actual arrival time for a postman p to arrive at destination i
    penalty={}
    #g.finddistance is equivalent to OD between origin and destinations, can have arbitrary directions
    V = set(range(NN))
    
    for p in postmen:
        for o in range(len(unassignedorders)):
            distance1=g.findOD(postmeninitiallocation[int(p)],unassignedorders[int(o)])
            ODPO[int(p),int(o)]=distance1

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',if_output_solutions)
        env.start()
        with gp.Model(env=env) as m:
            for i in range(NN):
                penalty[i] = m.addVar(vtype=GRB.CONTINUOUS, name='P_%s' % (i))
                for p in range(PN):
                    AT[p,i] = m.addVar(vtype=GRB.CONTINUOUS, name="AT_%s_%d" % (p, i))
                    x2[p,i] = m.addVar(vtype=GRB.BINARY, name="x_%s_%d" % (p, i))
                    for j in range(NN):
                        x1[p, i, j] = m.addVar(vtype=GRB.BINARY, name="x_%s_%d_%d" % (p, i, j))
            m.setObjective(quicksum(x2[p, i] * ODPO[p, i] for i in range(NN) for p in range(PN))
            +quicksum(x1[p,i,j]*g.findOD(int(unassignedorders[i]), int(unassignedorders[j])) for p in range(PN)
                   for i in range(NN) for j in range(NN))+quicksum(penalty[i] for i in range(NN)))
            #reduce the time windows violation, reduce the travel time to the first node, reduce the travel time over the entire routes
    # every postman can only travel to one next destination or no
            for p in range(PN):
                m.addConstr(quicksum(x2[p,i] for i in range(NN))<=1)
                m.addConstr(quicksum(x2[p,i] for i in range(NN))*NN>=quicksum(x1[p,i,j] for i in range(NN) for j in range(NN)))
            for j in range(NN): #there is no central depot needed to return
                m.addConstr(quicksum(x2[p,j] for p in range(PN))+quicksum(x1[p,i,j] for i in V-{j} for p in range(PN))==1)
            for i in range(NN):
                m.addConstr(penalty[i] >= 0)
                for j in range(NN):#this order variable t should not exist if there is no assignment
                    for p in range(PN): #arrival time or service start time
                        m.addConstr(AT[p,i]+deliverytime+g.findOD(int(unassignedorders[i]),int(unassignedorders[j]))/speed
                        -M*(1-x1[p,i,j]) <= AT[p,j]) #after finishing delivery, the vehicle goes to another location and start
                for p in range(PN):#cannot go from itself to itself
                    m.addConstr(x2[p, i] + quicksum(x1[p, j, i] for j in range(NN)) >= quicksum(x1[p, i, j] for j in range(NN)))
                    m.addConstr(x1[p,i,i]==0)
                    m.addConstr(AT[p,i] <= M*(quicksum(x1[p,j,i] for j in range(NN))+x2[p,i])) #if this postman not assigned this mission, then zero
                    m.addConstr(penalty[i] >= AT[p, i]-timewindows[i])
                    m.addConstr(AT[p,i] >= x2[p,i]*ODPO[p,i]/speed)
            m.Params.MIPGap = 0
            m.optimize()
            if m.status == GRB.status.OPTIMAL:
                x1 = m.getAttr('x', x1)
                x2 = m.getAttr('x', x2)
                AT = m.getAttr('x', AT)
                postmantrajectory=[[] for _ in range(PN)]
                postmanarrivaltimes=[[] for _ in range(PN)]
                trajectorytimewindows=[[] for _ in range(PN)]
                if m.status == GRB.Status.OPTIMAL:
                    for i in range(NN):
                        for p in range(PN):
                            if x2[p,i]>0.9:
                                postmantrajectory[p].append(unassignedorders[i])
                                postmanarrivaltimes[p].append(AT[p,i])
                                trajectorytimewindows[p].append(timewindows[i])
                            for j in range(NN):
                                if x1[p,i,j]>0.9:
                                    postmantrajectory[p].append(unassignedorders[j])
                                    postmanarrivaltimes[p].append(AT[p,j])
                                    trajectorytimewindows[p].append(timewindows[j])
  #  print('totalorders',sum(x2[p,i] for p in range(PN) for i in range(NN))+sum(x1[p,i,j] for p in range(PN) for i in range(NN) for j in range(NN)))
    return postmantrajectory, postmanarrivaltimes, trajectorytimewindows

#------------------------Testing Module-----------------
def RecursiveReoptimisation(PN,NN,graphseed,ep):
    starttime=time.time()
    device = torch.device('cpu')
    g1 = Env.Graph(PN)  # initialise a graph
    g1 = Env.graph_generator(g1, NN, randomseed=graphseed, graphname=None)
    state = Env.State(PN, NN, device)
    g1, state.order_popuptime, state.order_stay_time, state.order_node, state.postman_prev_node, state.postman_xy \
                        , state.order_indice, state.order_timewindow = Env.random_generator(g1, lb, hb, NN, PN, state, randomseed=ep)
    state.Done=False
    averagedeliverytime=np.mean(state.order_stay_time)
    prevedge=[[state.postman_prev_node[i], state.postman_prev_node[i]] for i in range(PN)]
    postmen=np.linspace(0,PN-1,PN) #for initial state all postmen are unselected
    #unassignedorders=state.unassignedorder
    #timewindows=[state.unassignedtimewindow[i]+state.unassignedpopuptime[i] for i in range(len(state.unassignedtimewindow))]
    #deliverytime=state.unassignedstayingtime
    for i in range(PN):
        g1.postman_percentage[i]=0
    postmeninitiallocation=[prevedge[i][1] for i in range(PN)]
    #------------------------------------preparation steps------------------------
    state.postman_destination_node=[None for i in range(PN)]
    postmantrajectory=[[] for i in range(PN)]
    trajectorytimewindows=[[] for i in range(PN)]
    currentgap=0
    counter=0
    state.elapse=0 #no elapse at the initial time step]
    state.postman_destination_indice=[None for _ in range(PN)]
    alldelivered=0
    if display_details:
        print('totalorders', len(state.order_node))
    #------------------------------------start iteration ------------------------
    while state.Done!=True:
        state.postman_elapse = [None for i in range(PN)]
       # if state.elapse == totaltime:
       #     if len(state.unassignedorder) == 0 and postmantrajectory==[[] for _ in range(PN)]:  # all assignments are made
      #          print('criteriareached')
      #          state.Done = True
        if state.current_time+state.elapse>=currentgap: #deadline triggered, need a re-optimisation, this should never be the initial case
            #print('enter reoptimisation')
            gapelapse = currentgap - state.current_time
            #print(currentgap,state.currenttime,state.elapse)
            currentgap+=gap
            if counter==0:  #if initial, no need to elapse
                state.elapse = 0
                for i in range(len(state.order_node)):
                    if state.order_popuptime[i] ==0:
                        state.unassigned_order_node = state.unassigned_order_node + [int(state.order_node[i])]
                        state.unassigned_order_indice = state.unassigned_order_indice + [int(state.order_indice[i])]
                        state.unassigned_order_popuptime = state.unassigned_order_popuptime + [state.order_popuptime[i]]
                        state.unassigned_order_staytime = state.unassigned_order_staytime + [state.order_stay_time[i]]
                        state.unassigned_order_timewindow = state.unassigned_order_timewindow + [state.order_timewindow[i]]
            else:
                state.elapse = gapelapse
                state.current_time = gapelapse + state.current_time
                for i in range(len(state.order_node)):
                    if state.current_time >= state.order_popuptime[i] > state.previous_time:
                        state.unassigned_order_node = state.unassigned_order_node + [int(state.order_node[i])]
                        state.unassigned_order_indice = state.unassigned_order_indice + [int(state.order_indice[i])]
                        state.unassigned_order_popuptime = state.unassigned_order_popuptime + [state.order_popuptime[i]]
                        state.unassigned_order_staytime = state.unassigned_order_staytime + [state.order_stay_time[i]]
                        state.unassigned_order_timewindow = state.unassigned_order_timewindow + [state.order_timewindow[i]]
            for i in range(PN):
                if state.postman_destination_node[i]!=None:
                    postmeninitiallocation[i]=state.postman_destination_node[i]
                else: #the postman may remain stationary somewhere, therefore we take the postman initial node
                    postmeninitiallocation[i]=state.postman_prev_node[i]
            if display_details:
                print(state.unassigned_order_node)
            if state.unassigned_order_node==[]: #at gap time, no new order emerges
                pass
            else:
                unassignedorders = state.unassigned_order_node
                timewindows = [state.unassigned_order_timewindow[i] + state.unassigned_order_popuptime[i] for i in
                               range(len(state.unassigned_order_timewindow))]
                postmantrajectory, postmanarrivaltimes, trajectorytimewindows \
                    = VRTPW(g1, postmeninitiallocation, postmen, unassignedorders, timewindows, averagedeliverytime)
                if len(timewindows)!=len(flatten(postmantrajectory)) or len(timewindows)!=len(unassignedorders):
                    print(len(timewindows))
                    print(len(flatten(postmantrajectory)))
                    print('error wrong output size')
                    sys.exit()
                for i in range(PN):
                    if postmantrajectory[i] != [] and state.postman_destination_node[i]==None: #only trigger if postman destination is arrived
                        assignedorder=postmantrajectory[i][0]
                        state.postman_destination_node[i] = assignedorder
                        del postmantrajectory[i][0]
                        del trajectorytimewindows[i][0]
                        assignedindex=state.unassigned_order_node.index(assignedorder)
                        state.postman_destination_indice[i]=state.unassigned_order_indice[assignedindex]
                        distance1, _ = g1.find_distance(start=state.postman_prev_node[i], end=state.postman_destination_node[i])
                        g1.add_assignment(i, int(state.unassigned_order_node[assignedindex]),
                                          state.current_time, distance1, state.unassigned_order_popuptime[assignedindex],
                                          state.unassigned_order_indice[assignedindex],
                                          state.unassigned_order_timewindow[assignedindex], 'VRPTWOptimiser')
                        state.postman_elapse[i]=0
                        if display_details:
                            print('assign postman', i, 'to', state.unassigned_order_node[assignedindex], 'at', state.current_time, 'distance', distance1)
                        alldelivered+=1
                        del state.unassigned_order_node[assignedindex]
                        del state.unassigned_order_indice[assignedindex]
                        del state.unassigned_order_popuptime[assignedindex]
                        del state.unassigned_order_staytime[assignedindex]
                        del state.unassigned_order_timewindow[assignedindex]
        else:
            #state.currenttime += state.elapse
          #  print('continuation')
            for i in range(len(state.order_node)):
                if state.current_time >= state.order_popuptime[i] > state.previous_time:
                    state.unassigned_order_node = state.unassigned_order_node + [int(state.order_node[i])]
                    state.unassigned_order_indice = state.unassigned_order_indice + [int(state.order_indice[i])]
                    state.unassigned_order_popuptime = state.unassigned_order_popuptime + [state.order_popuptime[i]]
                    state.unassigned_order_staytime = state.unassigned_order_staytime + [state.order_stay_time[i]]
                    state.unassigned_order_timewindow = state.unassigned_order_timewindow + [state.order_timewindow[i]]
            for i in range(PN):
                if postmantrajectory[i] != [] and state.postman_destination_node[i] == None:
                    # only trigger if postman destination is arrived
                    assignedorder = postmantrajectory[i][0]
                    state.postman_destination_node[i] = postmantrajectory[i][0]
                    assignedindex = state.unassigned_order_node.index(assignedorder)
                    del postmantrajectory[i][0]
                    del trajectorytimewindows[i][0]
                    state.postman_destination_indice[i] = state.unassigned_order_indice[assignedindex]
                    distance1, _ = g1.find_distance(start=state.postman_prev_node[i], end=state.postman_destination_node[i])
                    g1.add_assignment(i, int(state.unassigned_order_node[assignedindex]),
                                      state.current_time, distance1, state.unassigned_order_popuptime[assignedindex],
                                      state.unassigned_order_indice[assignedindex],
                                      state.unassigned_order_timewindow[assignedindex], 'VRPTWOptimiser')
                    state.elapse=min(state.elapse, distance1 / speed + state.unassigned_order_staytime[assignedindex])
                    state.postman_elapse[i] = 0
                    if display_details:
                        print('assign postman', i, 'to', state.unassigned_order_node[assignedindex], 'at', state.current_time,
                          'distance', distance1)
                    alldelivered += 1
                    # -----------------------statistics--------------------
                    del state.unassigned_order_node[assignedindex]
                    del state.unassigned_order_indice[assignedindex]
                    del state.unassigned_order_popuptime[assignedindex]
                    del state.unassigned_order_staytime[assignedindex]
                    del state.unassigned_order_timewindow[assignedindex]
                # if after optimisation, still one or more postmen have no destination assigned, then we can ignore (insufficient orders)
        if counter==0:
            g1, state = Env.Environment(g1, state, initial=True)
        else:
            g1, state = Env.Environment(g1, state, initial=False)
        counter+=1
        state.previous_time=state.current_time
        state.elapse=min(state.elapse, currentgap - state.current_time)
        state.current_time = state.current_time + state.elapse
        
    postmanassignedtasks, postmanassignedtime, postmanassigneddistance, \
    customerpopuptime, postmanassignedindex, alltimewindows, assignmentmethod = g1.get_assigned()
    postmandeliveredtasks, postmandeliveredtimes, postmanorderindex = g1.get_delivered()
    for i in range(PN):
        postmanassignedtasks[i] = sort_together([postmanassignedindex[i], postmanassignedtasks[i]])[1]
        postmanassignedtime[i] = sort_together([postmanassignedindex[i], postmanassignedtime[i]])[1]
        customerpopuptime[i] = sort_together([postmanassignedindex[i], customerpopuptime[i]])[1]
        postmanassigneddistance[i] = sort_together([postmanassignedindex[i], postmanassigneddistance[i]])[1]
        postmanassignedindex[i] = sort_together([postmanassignedindex[i], postmanassignedindex[i]])[1]
        assignmentmethod[i] = sort_together([postmanassignedindex[i], assignmentmethod[i]])[1]
        postmandeliveredtimes[i] = sort_together([postmanassignedindex[i], postmandeliveredtimes[i]])[1]
        postmanorderindex[i] = sort_together([postmanassignedindex[i], postmanorderindex[i]])[1]
    
    episodereward = Env.episode_reward(allpopuptimes=customerpopuptime,
                                       postmanassignedindex=postmanassignedindex,
                                       alldeliveredtimes=postmandeliveredtimes,
                                       postmanorderindex=postmanorderindex,
                                       alltimewindows=alltimewindows)
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
    return df, episodereward
    