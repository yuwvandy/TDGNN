from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Linear
import torch

class prop_sum(MessagePassing):
    def __init__(self, num_classes, layers, **kwargs):
        super(prop_sum, self).__init__(aggr = 'add', **kwargs)
        self.layers = layers

    def forward(self, x, edge_index, edge_weight):
        embed_layer = []
        embed_layer.append(x)

        if(self.layers != [0]):
            for layer in self.layers:
                # edge_weight[layer - 1] = edge_weight[layer - 1]/torch.sum(edge_weight[layer - 1])
                h = self.propagate(edge_index[layer - 1], x = x, norm = edge_weight[layer - 1])
                embed_layer.append(h)

        embed_layer = torch.stack(embed_layer, dim = 1)

        return embed_layer

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def reset_parameters(self):
        pass


class prop_weight(MessagePassing):
    def __init__(self, num_classes, layers, **kwargs):
        super(prop_weight, self).__init__(aggr = 'add', **kwargs)

        self.weight = torch.nn.Parameter(torch.ones(len(layers) + 1), requires_grad = True)

        self.layers = layers

    def forward(self, x, edge_index, edge_weight):
        embed_layer = []
        embed_layer.append(self.weight[0] * x)

        for i in range(len(self.layers)):
            h = self.propagate(edge_index[self.layers[i] - 1], x = x, norm = edge_weight[self.layers[i] - 1])
            embed_layer.append(self.weight[i + 1] * h)


        embed_layer = torch.stack(embed_layer, dim = 1)

        return embed_layer

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def reset_parameters(self):
        self.weight = torch.nn.Parameter(torch.ones(len(self.layers) + 1), requires_grad = True)



class TDGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(TDGNN, self).__init__()
        self.args = args
        self.agg = args.agg

        self.lin1 = Linear(dataset.num_features, self.args.hidden)
        self.lin2 = Linear(self.args.hidden, dataset.num_classes)

        if(self.agg == 'sum'):
            self.prop = prop_sum(dataset.num_classes, self.args.layers)
        if(self.agg == 'weighted_sum'):
            self.prop = prop_weight(dataset.num_classes, self.args.layers)


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p = self.args.dropout1, training = self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p = self.args.dropout2, training = self.training)
        x = self.lin2(x)

        x = self.prop(x, self.args.hop_edge_index, self.args.hop_edge_att)
        x = torch.sum(x, dim = 1)

        return F.log_softmax(x, dim = 1)
