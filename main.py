import argparse
from dataset import *
from learn import *
from model import *
from utils import *
from os import path
from tqdm import tqdm
import random
from torch import tensor


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--wd1', type=float, default=0.006)
parser.add_argument('--wd2', type=float, default=0.006)
parser.add_argument('--wd3', type=float, default=0)
parser.add_argument('--early_stopping', type=int, default=0)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout1', type=float, default=0.8)
parser.add_argument('--dropout2', type=float, default=0.8)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--tree_layer', type=int, default=10)
parser.add_argument('--layers', nargs='+', type = int)
parser.add_argument('--setting', type=str, default='semi')
parser.add_argument('--shuffle', type=str, default='random')
parser.add_argument('--agg', type=str, default='sum')
parser.add_argument('--tree_decompose', type=bool, default=True)


args = parser.parse_args()

#import the dataset
dataset = get_dataset(args.dataset, args.normalize_features)
data = dataset[0]

#split the dataset
if(args.setting == 'semi'):
    data_splitting = random_planetoid_splits
elif(args.setting == 'full'):
    data_splitting = random_full_splits

#run tree decomposition
edge_file = './tree_info/hop_edge_index_' + args.dataset + '_' + str(args.tree_layer)
if(path.exists(edge_file) == False):
    edge_info(dataset, args)

#load tree decomposed edge_index and multi-hop edge weight(att)
hop_edge_index = torch.load('./tree_info/hop_edge_index_' + args.dataset + '_' + str(args.tree_layer))
hop_edge_att = torch.load('./tree_info/hop_edge_att_' + args.dataset + '_' + str(args.tree_layer))

acc = np.zeros(args.runs, dtype=float)
pbar = tqdm(range(args.runs), unit='run')
count = 0

for count in pbar:
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if(args.setting == 'semi'):
        if(args.shuffle == 'random'):
            data = data_splitting(data, dataset.num_classes)
        elif(args.shuffle == 'fix'):
            pass
    elif(args.setting == 'full'):
        data = data_splitting(data, dataset.num_classes, count%10, args.dataset)

    model = TDGNN(dataset, args).to(args.device)
    data = data.to(args.device)

    for layer in args.layers:
        hop_edge_index[layer - 1] = hop_edge_index[layer - 1].type(torch.LongTensor).to(args.device)
        hop_edge_att[layer - 1] = hop_edge_att[layer - 1].to(args.device)
    args.hop_edge_index = hop_edge_index
    args.hop_edge_att = hop_edge_att

    if(args.agg == 'sum'):
        optimizer = torch.optim.Adam([
                dict(params=model.lin1.parameters(), weight_decay=args.wd1),
                dict(params=model.lin2.parameters(), weight_decay=args.wd2)], lr=args.lr)
    elif(args.agg == 'weighted_sum'):
        optimizer = torch.optim.Adam([
                dict(params=model.lin1.parameters(), weight_decay=args.wd1),
                dict(params=model.lin2.parameters(), weight_decay=args.wd2),
                dict(params=model.prop.parameters(), weight_decay=args.wd3)], lr=args.lr)


    best_val_loss = float('inf')
    best_val_acc = 0.0
    test_acc = 0
    val_loss_history = []

    for epoch in range(0, args.epochs):
        out = train(model, optimizer, data)
        eval_info = evaluate(model, data)
        eval_info['epoch'] = epoch

        if eval_info['val_loss'] < best_val_loss:
            best_val_loss = eval_info['val_loss']
            test_acc = eval_info['test_acc']

        val_loss_history.append(eval_info['val_loss'])
        if args.early_stopping > 0 and epoch > args.epochs // 2:
            tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
            if eval_info['val_loss'] > tmp.mean().item():
                break

    acc[count] = test_acc

    # print('Acc:', np.mean(acc[0:(count + 1)]), np.std(acc[0:(count + 1)]))

print('Acc:', np.mean(acc), 'Std:', np.std(acc))
