## Atari
RioRL code for Atari
## Env
please following the install instruction of 
1. https://github.com/kzl/decision-transformer 
2. https://github.com/takuseno/d3rlpy

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
python riorl_dt_exp.py --game Qbert
```

Conservative Q-Learning
```shell
python riorl_cql_exp.py --game Breakout
```

Behavior Cloning
```shell
python riorl_dt_exp.py --game Breakout
```

