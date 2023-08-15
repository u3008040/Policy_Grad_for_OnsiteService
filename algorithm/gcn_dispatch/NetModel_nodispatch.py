import torch
import torch.nn as nn
import utils.Parameters as Parameters
import time

global args
import math
from algorithm.gcn_dispatch.NetLayers import ResidualGatedGCNLayer
from torch.distributions import Categorical

args = Parameters.parameters()
import numpy as np

speed = vars(args)['speed']
global NN_lower
NN_lower = vars(args)['NNlower']
from more_itertools import sort_together
import sys
import numpy

display_details = 0
global exploration_prob
exploration_prob = vars(args)['exploration_prob']


def selector(Multinomial, softmax, postmannum, ordernum, Reject_NA_Rate, Greedyaction, Greedyprocess, banselection,
             device):  # multinomial is a binary indicator, True or False. softmax contains probabilities
    # softmax: batch * postman * orders
    if Greedyprocess:
        #  print('greedyprocess')
        selectedpostman = list(Greedyaction[0].keys())[0]
        selectedorder = Greedyaction[0][selectedpostman]
        equivalent_action = selectedpostman * (ordernum + 1) + selectedorder
        if equivalent_action in banselection:
            print('error, equivalent action in ban selection')
            raise KeyboardInterrupt
        action_distributions = Categorical(softmax)
        entropy = action_distributions.entropy()
        logprob = action_distributions.log_prob(torch.tensor(equivalent_action).to(device))
        # print(softmax)
        # print(torch.exp(logprob))
        return Greedyaction, logprob, entropy
    if display_details: print('softmax', softmax)
    if len(softmax[0]) == 1:
        print(postmannum)
        print(ordernum)
        print('err selector')
        sys.exit()
    if Multinomial:  # 概率采样；保证有探索
        # ------------------select the first one----------------
        # print('multinomial')
        selected = {0: {}}
        selectedpostmen = []
        selectedorders = []
        action_distributions = Categorical(softmax)
        entropy = action_distributions.entropy()
        if Greedyprocess != True and np.random.uniform() <= exploration_prob:
            selectionlist = np.arange(softmax.size(1))
            selectionlist1 = [i for i in selectionlist if i not in banselection]
            action = np.random.choice(selectionlist1)
            # print('random_exploration')
            logprob = action_distributions.log_prob(torch.tensor(action).to(device))
            remainder = int(action) % (ordernum + 1)  # equivalent to order index
            postmanindex = (int(action) - remainder) / (ordernum + 1)  # equivalent to postman index
            if remainder == ordernum:
                selectedpostmen.append(int(postmanindex))
                selectedorders.append(None)
                selected[0][int(postmanindex)] = None
            #  print('noassignment')
            else:
                # selectedorders.append(int(remainder))
                selected[0][int(postmanindex)] = int(remainder)
        else:
            
            action = torch.argmax(softmax, dim=1)
            # action = action_distributions.sample()
            logprob = action_distributions.log_prob(action)
            remainder = int(action) % (ordernum + 1)  # equivalent to order index
            postmanindex = (int(action) - remainder) / (ordernum + 1)  # equivalent to postman index
            if remainder == ordernum:
                selectedpostmen.append(int(postmanindex))
                selectedorders.append(None)
                selected[0][int(postmanindex)] = None
            #  print('noassignment')
            else:
                # selectedorders.append(int(remainder))
                selected[0][int(postmanindex)] = int(remainder)
        # -----------------select the following ---------------
        return selected, logprob, entropy
    else:  # 不训练的时候使用
        # print('not multinomial')
        selected = {0: {}}
        selectedpostmen = []
        selectedorders = []
        action = torch.argmax(softmax, dim=1)
        action_distributions = Categorical(softmax)
        logprob = torch.log(torch.max(softmax, dim=1)[0])
        remainder = int(action) % (ordernum + 1)  # equivalent to order index
        postmanindex = (int(action) - remainder) / (ordernum + 1)  # equivalent to postman index
        if remainder == ordernum:
            selectedpostmen.append(int(postmanindex))
            selectedorders.append(None)
            selected[0][int(postmanindex)] = None
            # if display_details:
            # print('noassignment')
        else:
            # selectedorders.append(int(remainder))
            selected[0][int(postmanindex)] = int(remainder)
        #  print(softmax)
        return selected, logprob, torch.zeros(1).to(device)


class ResidualGatedGCNModel(nn.Module):
    """Residual Gated GCN Model for outputting predictions as edge adjacency matrices.

    References:
        code:https://github.com/chaitjo/graph-convnet-tsp/blob/master/models/gcn_model.py
    """
    
    def __init__(self, is_training, PN, Mag_Factor):
        super(ResidualGatedGCNModel, self).__init__()
        # self.dtypeFloat = dtypeFloat
        # self.dtypeLong = dtypeLong
        #  self.PN=PN #number of postmen in a map
        # Define net parameters
        if (torch.cuda.is_available()):
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.Mag_Factor=Mag_Factor
        self.type(torch.float32)
        self.is_training = is_training
        self.node_dim = vars(args)['node_dim']
        self.voc_edges_in = vars(args)['voc_edges_in']
        self.hidden_dim = vars(args)['hidden_dim']
        self.postman_dim = vars(args)['postman_dim']
        self.num_layers = vars(args)['num_layers']
        self.aggregation = vars(args)['aggregation']
        self.edge_dim = vars(args)['edge_dim']
        self.embedding = nn.Embedding(PN, 3)
        # Node and edge embedding layers/lookups
        self.nodes_coord_embedding = nn.Linear(self.node_dim+NN_lower-1, self.hidden_dim,
                                               bias=True)  # converts node coordinates to embeddings
        # probably need to take into account previous order information in the nodal embeddings
        self.edges_values_embedding = nn.Linear(self.edge_dim, self.hidden_dim // 2,
                                                bias=True)  # convert edge weight to edge embedding
        self.edges_embedding = nn.Embedding(self.voc_edges_in, self.hidden_dim // 2)  # edge adjacency matrix
        # self.to_x=nn.Linear(self.hidden_dim,self.hidden_dim,bias=False)
        self.to_num = nn.Linear(self.hidden_dim, 1, bias=False)
        self.input_conversion = nn.Linear(self.node_dim + self.postman_dim + 2 + NN_lower - 1, self.hidden_dim,
                                          bias=False)
        self.postman_conversion=nn.Linear(self.postman_dim,self.hidden_dim,bias=True)
        # nn.embedding module converts individual word to specific embeddings.
        # the first index is the size of this dictionary. the second index is the embedding size of each word. For example,
        # there are 10 words in a dictionary, then 1 has an embedding, 2 has an embedding, 3 has an embedding etc. Each number
        # has an embedding of size (second index)
        # Define GCN Layers
        self.selfass = nn.Linear(self.hidden_dim*2, self.hidden_dim, bias=True)
        self.tanh = nn.Tanh()
        # Define MLP classifiers
        self.postman_emb = nn.Linear(self.postman_dim, self.hidden_dim, bias=True)
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm_emb = nn.LayerNorm(self.hidden_dim)
        self.globalconversion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        transformer_layers1 = []
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        for layer in range(self.num_layers):
            transformer_layers1.append(torch.nn.TransformerEncoderLayer(self.hidden_dim, nhead=4,
                                                                        dim_feedforward=self.hidden_dim, dropout=0,
                                                                        # activation=nn.LeakyReLU(),
                                                                        layer_norm_eps=1e-05, batch_first=True,
                                                                        device=self.device))
        self.transformer_layers1 = nn.ModuleList(transformer_layers1)
        transformer_layers2 = []
        for layer in range(self.num_layers):
            transformer_layers2.append(torch.nn.TransformerEncoderLayer(self.hidden_dim, nhead=4,
                                                                        dim_feedforward=self.hidden_dim, dropout=0,
                                                                        layer_norm_eps=1e-05, batch_first=True,
                                                                        device=self.device))
        self.transformer_layers2 = nn.ModuleList(transformer_layers2)
    
    def forward(self, all_adjacency_np, allOD_np, x_nodes_np, postman_np, if_assignment_required,
                if_postman_unassigned, prevemb,
                state, g, nodepostman, Reject_NA_Rate, Greedyprocess=False,
                NNassignment=None, NNlogprobs=None, Entropy=None, recursion=False, istraining=None,
                additional_penalty=0, batch=None):
        """
        Args:
            #x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
            #convention of the adjacency matrix, each loop is 2, each connection is 1, therefore only 0,1,2, are possible
            #x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
            #x_nodes_coord: Input node coordinates (batch_size, num_nodes, node_dim) should also include the delivery order information
            # node_cw: Class weights for nodes loss
            # postman_indices: which postman the delivery task should be assigned to. should be an index consistent with the order
            V: vertices, H: embedding
            #customer_indices: which customer (location indices) the delivery task can be assigned to
        Returns:
            choice: postman indices or customer indices
            if_customer: True: list is customers, False: list is for postmen
            prevemb: graph embeddings from the previous iteration
            ATOD: Agent Task OD matrix. The distance between each pair of agent and task, batch*postmen*locations
        """
        # start = time.time()
        #  print('enterRL')
        initialnode = state.postman_prev_node
        # Node and edge embedding
        x_edges = torch.from_numpy(all_adjacency_np).type(torch.LongTensor).to(self.device)  # adjacency matrices
        x_edges_values = torch.from_numpy(allOD_np).to(torch.float32).to(self.device)  # OD matrices
        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3)) / 4  # B x (NN+PN) x (NN+PN) x H/2  convert weights into embeddings
        e_tags = self.edges_embedding(x_edges) / 4
        e = torch.cat((e_vals, e_tags), dim=3)
        # problematic. why some rows are entirely zero
        # nodepostman is the postman at a specific location, input is location,output postman, initialnode is the initial location of a postman
        PN_total = len(if_postman_unassigned)
        NN_total = len(if_assignment_required)  # number of destinations to be delivered
        PN_unassigned = sum(if_postman_unassigned)
        NN_unassigned = sum(if_assignment_required)
        adjacency = x_edges - 2 * torch.tensor(np.identity(PN_total + NN_total), dtype=torch.float32).to(self.device)
        currentnode = []
        for i in range(PN_total):
            if if_postman_unassigned[i] == True:
                currentnode.append(int(initialnode[i]))
                if if_assignment_required[int(initialnode[i])] == True:
                    print('err, this node has been assigned')
                    sys.exit()
        #  customer_indices = [i for i in range(NN_total) if if_assignment_required[i]]
        postman_indices = [i for i in range(PN_total) if if_postman_unassigned[i]]
        locationindices = [i for i, j in enumerate(if_assignment_required) if j == True]
        postman_indices1 = [NN_total + i for i in postman_indices]
        if display_details: print('PN', PN_unassigned)
        if display_details: print('NN', NN_unassigned)
        if NN_unassigned == 0:
            print('error, wrong unassigned number')
            sys.exit()
        x_emb = np.zeros((1, NN_total, 2))
        alreadylisted = []
        embindex1 = []
        embindex2 = []
        for i in range(NN_total):
            if nodepostman[i] != []:  # if a postman is assigned to a specific location, put his embedding here
                x_emb[0, i, 0] = int(nodepostman[i][0])
                alreadylisted.append(i)
                embindex1.append(i)
                ##self.embedding(torch.tensor(nodepostman[i][0]).to(self.device)) / 2
                if len(nodepostman[i]) == 1:
                    pass
                else:
                    x_emb[0, i, 1] = int(nodepostman[i][1])
                    alreadylisted.append(i)
                    embindex2.append(i)
                    # self.embedding(torch.tensor(nodepostman[i][1]).to(self.device)) / 2
        for i in range(PN_total):  # if a postman idles at a location, put his embedding here
            if if_postman_unassigned[i] == True:
                if int(initialnode[i]) in alreadylisted:  # in alredy listed
                    x_emb[0, int(initialnode[i]), 1] = int(i)
                    embindex2.append(int(initialnode[i]))
                else:  # not have listed
                    x_emb[0, int(initialnode[i]), 0] = int(i)
                    alreadylisted.append(int(initialnode[i]))
                    embindex1.append(int(initialnode[i]))
        embindex1 = np.sort(embindex1)
        embindex2 = np.sort(embindex2)
        # x_nodes_coord = x_nodes_coord.to(torch.float32).to(self.device)  # coordinates
        postman_emb = self.embedding(torch.from_numpy(np.arange(PN_total)).to(self.device)).unsqueeze(0)
        node_emb1 = torch.zeros(1, NN_total, 3).to(self.device)
        node_emb2 = torch.zeros(1, NN_total, 3).to(self.device)
        if len(embindex1) != 0:
            node_emb1[0, embindex1, :] = \
                self.embedding(torch.from_numpy(x_emb[:, :, 0]).type(torch.LongTensor).to(self.device))[0, embindex1, :]
        if len(embindex2) != 0:
            node_emb2[0, embindex2, :] = \
                self.embedding(torch.from_numpy(x_emb[:, :, 1]).type(torch.LongTensor).to(self.device))[0, embindex2, :]
        node_emb_all = torch.cat((node_emb1, node_emb2), dim=2)
     #  postman_node_emb = torch.cat((postman_emb.unsqueeze(2).repeat(1, 1, NN_total, 1), node_emb_all), dim=-1)
        if istraining == None:
            Multinomial = self.is_training  # if training, then multinomial selector, if not training, then use a greedy selector.
        else:
            Multinomial = istraining
        batch_size = all_adjacency_np.shape[0]
        nearestnodes=[list(g.nearest3nodes[i]) for i in range(NN_lower)]
        p=self.postman_conversion(torch.cat((torch.from_numpy(postman_np).type(torch.float32).to(self.device)
                                             ,postman_emb),dim=-1))
        x2=self.nodes_coord_embedding(torch.cat((torch.from_numpy(x_nodes_np).type(torch.float32).to(self.device)
                        ,node_emb_all,torch.tensor(nearestnodes).unsqueeze(0).type(torch.float32).to(self.device)),dim=-1))
        x1=torch.cat((x2,p),dim=1)
        for layer in range(self.num_layers):
            x1, e = self.gcn_layers[layer](x1, e, adjacency, PN_total, NN_total)
            # B x V x H, B x V x V x H  each layer has different parameters
        a=x1[:,locationindices].unsqueeze(1).repeat(1,len(postman_indices),1,1)
        b=x1[:,postman_indices1].unsqueeze(2).repeat(1,1,len(locationindices),1)
        a1=x1[:,currentnode].unsqueeze(2)
        b1=x1[:,postman_indices1].unsqueeze(2)
        x2=torch.cat((torch.cat((b,a),dim=-1),torch.cat((b1,a1),dim=-1)),dim=2)
        attention2 = self.to_num(self.relu(self.selfass(x2)))
        if istraining: additional_penalty += sum([max(abs(k) - 2.5, 0) for k in attention2.view(1, -1).tolist()[0]])
        if additional_penalty > 0: print('warning, risk of explosion')
        if display_details: print('customer', locationindices, nodepostman)
        if display_details: print('postman', postman_indices, initialnode)
        if display_details: print(x_nodes_coord)
        if display_details: print(postman_feature)
        if NNassignment == None and NNlogprobs == None:
            NNassignment = {}
            NNlogprobs = {}
            Entropy = {}
            for i in range(batch_size):
                NNassignment[i] = {}
                NNlogprobs[i] = []
                Entropy[i] = []
        # if more than customer than postman, postman should of course choose customer
        # X dimension Batch * PN * NN * Emb
        # concatenation of chosen postmen embedding, customer embedding, distances between two locations and infeasibility matrix
        #print(PN_unassigned,NN_unassigned)
        #print('attention2',attention2)
        attention3 = 10 * self.tanh(attention2).view(1, -1)
        #print('attention3',attention3)
        rate = np.random.uniform()
        Reject_NA_Rate2 = Reject_NA_Rate
        banselection = []
        if rate < Reject_NA_Rate:  # ban selection
            for p_idx in range(PN_unassigned):
                banselection.append((1 + p_idx) * (1 + NN_unassigned) - 1)
            if Reject_NA_Rate == 1:
                if sum(if_postman_unassigned) == 1 or sum(if_assignment_required) == 1:
                    for k in range(attention3.size(-1)):
                        if k in banselection:
                            attention3[0, k] = torch.tensor(-float('inf')).to(self.device)
                    Reject_NA_Rate2 = 1
            else:
                select = np.argmax(attention3.tolist())
                if select in banselection:
                    for k in range(attention3.size(-1)):
                        if k in banselection:
                            attention3[0, k] = torch.tensor(-float('inf')).to(self.device)
        attention = self.softmax(attention3)
      #  print('attention',attention)
        if display_details: print('attention2', attention2)
        if display_details: print('attention', 10 * self.tanh(attention2.view(1, -1)))
        
        OD = np.zeros([PN_unassigned, NN_unassigned])
        for i in range(PN_unassigned):
            for j in range(NN_unassigned):
                D, _ = g.find_distance(start=initialnode[postman_indices[i]], end=locationindices[j])
                OD[i, j] = D
        actionGreedy = {0: {}}
        if display_details: print(OD)
        for j in range(batch_size):
            actionGreedy[j][numpy.where(OD == numpy.amin(OD))[0][0]] = int(numpy.where(OD == numpy.amin(OD))[1][0])
            action, logprob, entropy = selector(Multinomial, attention, PN_unassigned, NN_unassigned, Reject_NA_Rate2,
                                                actionGreedy, Greedyprocess, banselection, self.device)
            for i in action[j].keys():
                if action[j][i] != None:
                    NNassignment[j][postman_indices[i]] = \
                        locationindices[action[j][i]]
                else:
                    NNassignment[j][postman_indices[i]] = None
                NNlogprobs[j].append(logprob[j])
                Entropy[j].append(entropy[j])
            assignmentregistration = []
            if 1 < PN_unassigned and 1 < NN_unassigned:  # recurse to find all solutions
                # if there is only one remaining postman and one destination then we can make direct assignment
                for i in action[j].keys():
                    if action[j][i] != None:
                        postman_idx = int(postman_indices[i])
                        location_idx = int(locationindices[action[j][i]])
                        current_idx = int(initialnode[postman_indices[i]])
                        if postman_np[j, postman_idx, 4] != 0:
                            print('Netmodel error, this postman is working')
                            raise KeyboardInterrupt
                        postman_np[j, postman_idx, 4] = 1  # 1 for working
                        postman_np[j, postman_idx, 2] = x_nodes_np[j, location_idx, 0] /self.Mag_Factor # destination x coord
                        postman_np[j, postman_idx, 3] = x_nodes_np[j, location_idx, 1] /self.Mag_Factor # destination y coord
                        D, _ = g.find_distance(start=current_idx, end=location_idx)
                        postman_np[j, postman_idx, 6] = (D / speed) / 20  # expected arrival time
                        postman_np[j, postman_idx, 7] = (state.current_time - state.node_earliest_popuptime[
                            location_idx] - state.node_earliest_timewindow[
                                                             location_idx]) / 50  # destination time
                        x_nodes_np[j, location_idx, 2] -= 1 / PN_total
                        # how many unassigned orders at this location, assigned then remove
                        x_nodes_np[j, location_idx, 7] = (D / speed) / 20
                        if x_nodes_np[j, location_idx, 2] < 0:
                            print('error, mission number smaller than 0')
                            raise KeyboardInterrupt
                        # no need to care about the third dimension, because every single location is provided
                        # the earliest popup time
                        x_nodes_np[j, location_idx, 4] += 1 / PN_total
                        # how many postmen assigned to this location
                        # postman is no longer unassigned, move out from this location
                        assignmentregistration.append(postman_idx)
                        # add postman information to the new destination
                        nodepostman[int(location_idx)].append(postman_idx)  # assign this postman to this location
                        state.postman_destination_node[postman_idx] = location_idx
                        # otherwise it will be rewritten
                        if_assignment_required[location_idx] = 0
                    else:
                        pass
                    if_postman_unassigned[postman_indices[i]] = 0  # no longer require assignment
                for k in range(PN_total):
                    if if_postman_unassigned[k] == True:  # if unassigned add information
                        if k in assignmentregistration:
                            print('assignment registration conflict')
                            raise KeyboardInterrupt
                NNassignment, NNlogprobs, Entropy, prevemb, additional_penalty = ResidualGatedGCNModel.forward(self,
                                                                                                               all_adjacency_np,
                                                                                                               allOD_np,
                                                                                                               x_nodes_np,
                                                                                                               postman_np,
                                                                                                               if_assignment_required,
                                                                                                               if_postman_unassigned,
                                                                                                               prevemb,
                                                                                                               state, g,
                                                                                                               nodepostman,
                                                                                                               Reject_NA_Rate,
                                                                                                               Greedyprocess,
                                                                                                               NNassignment,
                                                                                                               NNlogprobs,
                                                                                                               Entropy,
                                                                                                               True,
                                                                                                               istraining,
                                                                                                               additional_penalty,
                                                                                                               batch=batch)
        if recursion:
            return NNassignment, NNlogprobs, Entropy, prevemb, additional_penalty
        else:
            NNlogprobs1 = {}
            Entropy1 = {}
            for i in range(batch_size):
                NNlogprobs1[i] = torch.stack(NNlogprobs[i]).sum()
                Entropy1[i] = torch.stack(Entropy[i]).sum()
            return NNassignment, NNlogprobs1, Entropy1, prevemb, additional_penalty
