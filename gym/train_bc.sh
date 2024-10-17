# Behavior Cloning (BC)
for seed in 0 1 2 3
do
    python train_dt.py --seed $seed --env hopper --dataset medium --model_type bc
done

for seed in 0 1 2 3
do
    python train_dt.py --seed $seed --env hopper --dataset medium-replay --model_type bc
done

for seed in 0 1 2 3
do
    python train_dt.py --seed $seed --env hopper --dataset medium-expert --model_type bc
done

for seed in 0 1 2 3
do
    python train_dt.py --seed $seed --env halfcheetah --dataset medium --model_type bc
done

for seed in 0 1 2 3
do
    python train_dt.py --seed $seed --env halfcheetah --dataset medium-replay --model_type bc
done

for seed in 0 1 2 3
do
    python train_dt.py --seed $seed --env halfcheetah --dataset medium-expert --model_type bc
done


for seed in 0 1 2 3
do
    python train_dt.py --seed $seed --env walker2d --dataset medium --model_type bc
done

for seed in 0 1 2 3
do
    python train_dt.py --seed $seed --env walker2d --dataset medium-replay --model_type bc
done

for seed in 0 1 2 3
do
    python train_dt.py --seed $seed --env walker2d --dataset medium-expert --model_type bc
done