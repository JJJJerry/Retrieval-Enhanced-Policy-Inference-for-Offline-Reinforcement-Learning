# OpenAI Gym MuJoCo

## Env
please following the instruction of 
1. https://github.com/kzl/decision-transformer 
2. https://github.com/aviralkumar2907/CQL

For retrieval
```
pip install faiss-gpu
```

## Training
Decision Transformer
```shell
sh train_dt.sh
```

Conservative Q-Learning
```shell
sh train_cql.sh
```

Behavior Cloning
```shell
sh train_bc.sh
```

## inference

Decision Transformer
```shell
python riorl_dt_exp.py --env hopper-medium-expert-v2
```

Conservative Q-Learning
```shell
python riorl_cql_exp.py --env halfcheetah-medium-expert-v2
```

Behavior Cloning
```shell
python riorl_bc_exp.py --env walker2d-meidum-expert-v2