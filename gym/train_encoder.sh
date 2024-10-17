# Encoder for retrieval
python train_dt.py --seed 789 --env hopper --dataset medium --model_type dt

python train_dt.py --seed 789 --env hopper --dataset medium-replay --model_type dt

python train_dt.py --seed 789 --env hopper --dataset medium-expert --model_type dt

python train_dt.py --seed 789 --env halfcheetah --dataset medium --model_type dt

python train_dt.py --seed 789 --env halfcheetah --dataset medium-replay --model_type dt

python train_dt.py --seed 789 --env halfcheetah --dataset medium-expert --model_type dt

python train_dt.py --seed 789 --env walker2d --dataset medium --model_type dt

python train_dt.py --seed 789 --env walker2d --dataset medium-replay --model_type dt

python train_dt.py --seed 789 --env walker2d --dataset medium-expert --model_type dt
