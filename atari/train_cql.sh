# Conservative Q-Learning (CQL)
for seed in 123 231 312 132
do
    python d3rl_train_cql.py --seed $seed --game 'Breakout'
done

for seed in 123 231 312 132
do
    python d3rl_train_cql.py --seed $seed --game 'Qbert' 
done

for seed in 123 231 312 132
do
    python d3rl_train_cql.py --seed $seed --game 'Pong' 
done

for seed in 123 231 312 132
do
    python d3rl_train_cql.py --seed $seed  --game 'Seaquest' 
done