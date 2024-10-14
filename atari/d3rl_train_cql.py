""" import d3rlpy
from mingpt.utils import set_seed
import argparse
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('--game', type=str, default='breakout')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda:1')
args=parser.parse_args()
game=args.game.lower()
seed=args.seed
set_seed(seed)

#with open(f'd3rl_dataset_cache/{game}.pkl','rb') as f:
#    dataset=pickle.load(f)
#with open(f'd3rl_dataset_cache/{game}_env.pkl','rb') as f:
#    env=pickle.load(f) 
dataset, env = d3rlpy.datasets.get_atari_transitions(
    game,
    fraction=0.01,
    index=0,
    num_stack=4,
)
# prepare algorithm
#cql = d3rlpy.algos.DiscreteCQLConfig(
#    observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
#    reward_scaler=d3rlpy.preprocessing.ClipRewardScaler(-1.0, 1.0),
#).create(device=args.device)
cql = d3rlpy.algos.DiscreteCQLConfig(
    learning_rate=5e-5,
    batch_size=32,
    optim_factory=d3rlpy.models.optimizers.AdamFactory(eps=1e-2 / 32),
    q_func_factory=d3rlpy.models.q_functions.QRQFunctionFactory(n_quantiles=200),
    observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
    target_update_interval=2000,
    gamma=0.99,
    alpha=4.0,
    reward_scaler=d3rlpy.preprocessing.ClipRewardScaler(-1.0, 1.0),
).create(device="cuda:0")

# start training
cql.fit(
    dataset,
    experiment_name=f'{game}_{seed}',
    n_steps=1000000,
    n_steps_per_epoch=10000,
    save_interval=1,
    evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
)

cql.save_model(f'cql_weights/{game}/cql_{game}_{seed}_.pt')
"""
import argparse

import d3rlpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="breakout")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device",type=str)
    args = parser.parse_args()

    d3rlpy.seed(args.seed)

    dataset, env = d3rlpy.datasets.get_atari_transitions(
        args.game,
        fraction=0.01,
        index=1 if args.game == "asterix" else 0,
        num_stack=4,
    )

    d3rlpy.envs.seed_env(env, args.seed)

    cql = d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=5e-5,
        optim_factory=d3rlpy.models.optimizers.AdamFactory(eps=1e-2 / 32),
        batch_size=32,
        alpha=4.0,
        q_func_factory=d3rlpy.models.q_functions.QRQFunctionFactory(
            n_quantiles=200
        ),
        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
        target_update_interval=2000,
        reward_scaler=d3rlpy.preprocessing.ClipRewardScaler(-1.0, 1.0),
    ).create(device=args.device)

    env_scorer = d3rlpy.metrics.EnvironmentEvaluator(env, epsilon=0.001)
    
    cql.fit(
        dataset,
        n_steps=50000000 // 4,
        n_steps_per_epoch=125000,
        evaluators={"environment": env_scorer},
        experiment_name=f"DiscreteCQL_{args.game}_{args.seed}",
    )
    cql.save_model(f'cql_weights/{args.game}/cql_{args.game}_{args.seed}_.pt')


if __name__ == "__main__":
    main()