import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import numpy as np
import sys


def Greedyoptimiser(OD,postmen,unassignedorderlen):
    
    assignment={}
    x={}
    #size1 postman
    #size2 destination
    size1,size2=OD.shape
    with gp.Env(empty=True) as env:
     #   print('optimisation triggered')
        env.setParam('OutputFlag',0)
        env.start()
        with gp.Model(env=env) as m:
            for i in range(size1):
                for j in range(size2):
                    x[i,j]=m.addVar(vtype=GRB.BINARY,name="x_%s_%d"%(i,j))
            m.setObjective(quicksum(x[i,j]*OD[i,j] for i in range(size1) for j in range(size2)))
            if len(postmen)>=unassignedorderlen:#if more postmen than needed
                for j in range(size2):
                    m.addConstr(quicksum(x[i,j] for i in range(size1))==1) #every  destination must have a postman assigned
                for i in range(size1):
                    m.addConstr(quicksum(x[i, j] for j in range(size2)) <= 1) #every postman must not have more than one location
            else: #if less postmen than needed
                for j in range(size2):
                    m.addConstr(quicksum(x[i,j] for i in range(size1))<=1) #every destination can only have one postman
                for i in range(size1):
                    m.addConstr(quicksum(x[i,j] for j in range(size2))==1) #every postman must have one destination
            m.Params.MIPGap = 0
            m.optimize()
            x = m.getAttr('x', x)
            selectedpostmen=[]
          #  print('postmen',postmen,'fromgreedyoptimiser')
            if m.status == GRB.Status.OPTIMAL:
                for i in range(size1):
                    for j in range(size2):
                        if x[i,j]==1:
                            selectedpostmen.append(i)
                            assignment[postmen[i]]=int(j)
            else:
                print('error,optimality not reached')
                sys.exit()
            if len(assignment)!=min(len(postmen),unassignedorderlen):
                print('error inconsistent optimisation result')
                sys.exit()
    return assignment,selectedpostmen