echo "=======Cora======="
echo "-----TDGNNs-----"
echo "-----semi-fixed-----"
python main.py --dataset='Cora' --epochs=3000 --early_stopping=0 --hidden=16 --lr=0.01 --wd1=0.006 --wd2=0.006 --dropout1=0.8 --dropout2=0.8 --K=10 --tree_layer=10 --setting='semi' --shuffle='fix' --agg='sum' --layers 1 2 3 4
echo "-----semi-random-----"
python main.py --dataset='Cora' --epochs=1500 --early_stopping=0 --hidden=32 --lr=0.01 --wd1=0.007 --wd2=0.007 --dropout1=0.8 --dropout2=0.8 --K=10 --tree_layer=10 --setting='semi' --shuffle='random' --agg='sum' --layers 1 2 3 4 5 6
echo "-----full-random-----"
python main.py --dataset='Cora' --epochs=300 --early_stopping=0 --hidden=32 --lr=0.01 --wd1=0.001 --wd2=0.004 --dropout1=0.5 --dropout2=0.8 --K=10 --tree_layer=10 --setting='full' --shuffle='random' --agg='sum' --layers 1 2 3 4

echo "-----TDGNNw-----"
echo "-----semi-fixed-----"
python main.py --dataset='Cora' --epochs=3000 --early_stopping=500 --hidden=32 --lr=0.01 --wd1=0.006 --wd2=0.007 --wd3=0.01 --dropout1=0.8 --dropout2=0.8 --K=10 --tree_layer=10 --setting='semi' --shuffle='fix' --agg='weighted_sum' --layers 1 2 3 4
echo "-----semi-random-----"
python main.py --dataset='Cora' --epochs=1500 --early_stopping=0 --hidden=32 --lr=0.01 --wd1=0.007 --wd2=0.007 --wd3=0.01 --dropout1=0.8 --dropout2=0.8 --K=10 --tree_layer=10 --setting='semi' --shuffle='random' --agg='weighted_sum' --layers 1 2 3 4 5 6
echo "-----full-random-----"
python main.py --dataset='Cora' --epochs=600 --early_stopping=100 --hidden=64 --lr=0.01 --wd1=0.005 --wd2=0.005 --wd3=0.001 --dropout1=0.5 --dropout2=0.8 --K=10 --tree_layer=10 --setting='full' --shuffle='random' --agg='weighted_sum' --layers 1 2 3 4 5


echo "=======Citeseer======="
echo "-----TDGNNs-----"
echo "-----semi-fixed-----"
python main.py --dataset='Citeseer' --epochs=4000 --early_stopping=0 --hidden=64 --lr=0.01 --wd1=0.02 --wd2=0.03 --dropout1=0.5 --dropout2=0.5 --K=10 --tree_layer=10 --setting='semi' --shuffle='fix' --agg='sum' --layers 1 2 3 4 5 6 7 8
echo "-----semi-random-----"
python main.py --dataset='Citeseer' --epochs=4000 --early_stopping=0 --hidden=64 --lr=0.01 --wd1=0.02 --wd2=0.03 --dropout1=0.5 --dropout2=0.5 --K=10 --tree_layer=10 --setting='semi' --shuffle='random' --agg='sum' --layers 1 2 3 4 5 6 7 8
echo "-----full-random-----"
python main.py --dataset='Citeseer' --epochs=4000 --early_stopping=0 --hidden=64 --lr=0.01 --wd1=0.02 --wd2=0.03 --dropout1=0.5 --dropout2=0.5 --K=10 --tree_layer=10 --setting='full' --shuffle='random' --agg='sum' --layers 1 2 3 4 5 6 7 8

echo "-----TDGNNw-----"
echo "-----semi-fixed-----"
python main.py --dataset='Citeseer' --epochs=2000 --early_stopping=0 --hidden=64 --lr=0.01 --wd1=0.02 --wd2=0.02 --wd3=0.05 --dropout1=0.5 --dropout2=0.5 --K=10 --tree_layer=10 --setting='semi' --shuffle='fix' --agg='weighted_sum' --layers 1 2 3 4 5 6 7 8
echo "-----semi-random-----"
python main.py --dataset='Citeseer' --epochs=3000 --early_stopping=500 --hidden=64 --lr=0.01 --wd1=0.01 --wd2=0.03 --wd3=0.05 --dropout1=0.5 --dropout2=0.5 --K=10 --tree_layer=10 --setting='semi' --shuffle='random' --agg='weighted_sum' --layers 1 2 3 4 5 6 7 8
echo "-----full-random-----"
python main.py --dataset='Citeseer' --epochs=500 --early_stopping=0 --hidden=128 --lr=0.01 --wd1=0.0005 --wd2=0.0001 --wd3=0.02 --dropout1=0.5 --dropout2=0.5 --K=10 --tree_layer=10 --setting='full' --shuffle='random' --agg='weighted_sum' --layers 1 2

echo "=======Cornell======="
echo "-----TDGNNs-----"
echo "-----full-random-----"
python main.py --dataset='Cornell' --epochs=1000 --early_stopping=200 --hidden=128 --lr=0.01 --wd1=0.001 --wd2=0.001 --dropout1=0 --dropout2=0 --K=10 --tree_layer=10 --setting='full' --shuffle='random' --agg='sum' --layers 5

echo "-----TDGNNw-----"
echo "-----full-random-----"
python main.py --dataset='Cornell' --epochs=1000 --early_stopping=200 --hidden=128 --lr=0.01 --wd1=0.001 --wd2=0.001 --wd3=0.001 --dropout1=0.5 --dropout2=0.5 --K=10 --tree_layer=10 --setting='full' --shuffle='random' --agg='weighted_sum' --layers 2 3 4 5 6

echo "=======Texas======="
echo "-----TDGNNs-----"
echo "-----full-random-----"
python main.py --dataset='Texas' --epochs=1000 --early_stopping=200 --hidden=128 --lr=0.01 --wd1=0.0005 --wd2=0.0005 --dropout1=0 --dropout2=0 --K=10 --tree_layer=10 --setting='full' --shuffle='random' --agg='sum' --layers 4 5

echo "-----TDGNNw-----"
echo "-----full-random-----"
python main.py --dataset='Texas' --epochs=1000 --early_stopping=200 --hidden=128 --lr=0.01 --wd1=0.0005 --wd2=0.0005 --wd3=0.0005 --dropout1=0.5 --dropout2=0.5 --K=10 --tree_layer=10 --setting='full' --shuffle='random' --agg='weighted_sum' --layers 2

echo "=======Wisconsin======="
echo "-----TDGNNs-----"
echo "-----full-random-----"
python main.py --dataset='Wisconsin' --epochs=500 --early_stopping=200 --hidden=128 --lr=0.01 --wd1=0.0001 --wd2=0.0001 --dropout1=0.5 --dropout2=0.5 --K=10 --tree_layer=10 --setting='full' --shuffle='random' --agg='sum' --layers 4 5

echo "-----TDGNNw-----"
echo "-----full-random-----"
python main.py --dataset='Wisconsin' --epochs=500 --early_stopping=200 --hidden=128 --lr=0.01 --wd1=0.0001 --wd2=0.0001 --wd3=0.0005 --dropout1=0.5 --dropout2=0.5 --K=10 --tree_layer=10 --setting='full' --shuffle='random' --agg='weighted_sum' --layers 3 4 5

echo "=======Actor======="
echo "-----TDGNNs-----"
echo "-----full-random-----"
python main.py --dataset='Actor' --epochs=500 --early_stopping=200 --hidden=32 --lr=0.01 --wd1=0.001 --wd2=0.001 --dropout1=0 --dropout2=0.8 --K=10 --tree_layer=10 --setting='full' --shuffle='random' --agg='sum' --layers 3

echo "-----TDGNNw-----"
echo "-----full-random-----"
python main.py --dataset='Actor' --epochs=500 --early_stopping=200 --hidden=32 --lr=0.01 --wd1=0.005 --wd2=0.005 --wd3=0.001 --dropout1=0 --dropout2=0.5 --K=10 --tree_layer=10 --setting='full' --shuffle='random' --agg='weighted_sum' --layers 3 4
