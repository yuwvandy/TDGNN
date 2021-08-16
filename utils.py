import networkx as nx
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter
import torch
import numpy as np


def edgelist2graph(edge_index, nodenum):
    """ Preprocess to get the graph data wanted
    """
    edge_index = edge_index.cpu().detach().numpy()
    adjlist = {i: [] for i in range(nodenum)}

    for i in range(len(edge_index[0])):
        adjlist[edge_index[0][i]].append(edge_index[1][i])


    return adjlist, nx.adjacency_matrix(nx.from_dict_of_lists(adjlist)).toarray()



def edge_info(dataset, args):
    adjlist, adjmatrix = edgelist2graph(dataset.data.edge_index, dataset.data.x.size(0))
    hop_edge_index, hop_edge_att = tree_decomposition(dataset.data.x.size(0), args.tree_layer, adjlist, dataset.data.edge_index)
    torch.save(hop_edge_index, './tree_info/hop_edge_index_' + args.dataset + '_' + str(args.tree_layer))
    torch.save(hop_edge_att, './tree_info/hop_edge_att_' + args.dataset + '_' + str(args.tree_layer))



def propagate(x, edge_index):
    """ feature propagation procedure: sparsematrix
    """
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes = x.size(0))

    #calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype = x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    edge_weight = deg_inv_sqrt[row]*deg_inv_sqrt[col] #for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]

    out = edge_weight.view(-1, 1)*x[row] #normalize the features on the starting point of the edge

    return scatter(out, edge_index[-1], dim = 0, dim_size = x.size(0), reduce = 'add')



def weight_deg(nodenum, edge_index, K):
    att = []


    x = torch.eye(nodenum)
    for i in range(K):
        x = propagate(x, edge_index)
        att.append(x)

    return att



def tree_decomposition(nodenum, K, adjlist, edge_index):
    #calculate the attention, the weight of each edge
    att = weight_deg(nodenum, edge_index, K)

    #to save the space, we use torch.tensor instead of list to save the edge_index
    hop_edge_index = [np.zeros((2, nodenum**2), dtype = int) for i in range(K)] #at most nodenum**2 edges
    hop_edge_att = [np.zeros(nodenum**2) for i in range(K)]
    hop_edge_pointer = np.zeros(K, dtype = int)


    for i in range(nodenum):
        hop_edge_index, hop_edge_att, hop_edge_pointer = BFS_adjlist(adjlist, nodenum, hop_edge_index, hop_edge_att, hop_edge_pointer, att, source = i, depth_limit = K)


    for i in range(K):
        hop_edge_index[i] = hop_edge_index[i][:, :hop_edge_pointer[i]]
        hop_edge_att[i] = hop_edge_att[i][:hop_edge_pointer[i]]

        hop_edge_index[i] = torch.tensor(hop_edge_index[i], dtype = torch.long)
        hop_edge_att[i] = torch.tensor(hop_edge_att[i], dtype = torch.float)

    return hop_edge_index, hop_edge_att



def BFS_adjlist(adjlist, nodenum, hop_edge_index, hop_edge_att, hop_edge_pointer, deg_att, source, depth_limit):
    visited = {}
    for node in adjlist.keys():
        visited[node] = 0

    queue, output = [], []
    queue.append(source)
    visited[source] = 1
    level = 1

    #initialize the edge pointed to the source node itself
    for i in range(len(hop_edge_index)):
        hop_edge_index[i][0, hop_edge_pointer[i]] = source
        hop_edge_index[i][1, hop_edge_pointer[i]] = source

    tmp = 0
    for k in range(0, depth_limit):
        tmp += deg_att[k][source, source]
    hop_edge_att[0][hop_edge_pointer[0]] = tmp
    hop_edge_pointer[0] += 1

    while queue:
        level_size = len(queue)
        while(level_size != 0):
            vertex = queue.pop(0)
            level_size -= 1
            for vrtx in adjlist[vertex]:
                if(visited[vrtx] == 0):
                    queue.append(vrtx)
                    visited[vrtx] = 1

                    hop_edge_index[level - 1][0, hop_edge_pointer[level - 1]] = source #distance = 1 is the first group in the hop_edge_list
                    hop_edge_index[level - 1][1, hop_edge_pointer[level - 1]] = vrtx
                    tmp = 0
                    for k in range((level - 1), depth_limit):
                        tmp += deg_att[k][source, vrtx]
                    hop_edge_att[level - 1][hop_edge_pointer[level - 1]] = tmp
                    hop_edge_pointer[level - 1] += 1

        level += 1
        if(level > depth_limit):
            break

    return hop_edge_index, hop_edge_att, hop_edge_pointer
