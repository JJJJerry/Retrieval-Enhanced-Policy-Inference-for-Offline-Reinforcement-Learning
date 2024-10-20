import argparse
import d3rlpy
import os

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="breakout")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device",type=str, default='cuda:0')
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
    save_dir=os.path.join('cql_weights',args.game)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name=f'cql_{args.game}_{args.seed}_pt'

    cql.save_model(os.path.join(save_dir,save_name))


if __name__ == "__main__":
    main()