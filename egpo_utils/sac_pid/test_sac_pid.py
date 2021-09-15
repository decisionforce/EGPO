from drivingforce.safety.sac_pid.sac_pid import SACPIDTrainer
from drivingforce.train import train, get_train_parser
from drivingforce.safety.sac_lag.train_sac_lag import Env

if __name__ == '__main__':
    args = get_train_parser().parse_args()

    exp_name = args.exp_name or "TEST"
    stop = int(1000000)

    config = dict(
        env=Env,
        env_config=dict(
        ),

        # ===== Training =====
        # recent_episode_num=2,
        optimization=dict(actor_learning_rate=1e-4, critic_learning_rate=1e-4, entropy_learning_rate=1e-4),
        prioritized_replay=True,
        horizon=1000,  # <<< For testing only!
        target_network_update_freq=1,
        timesteps_per_iteration=500,  # <<< For testing only!
        learning_starts=500,  # <<< For testing only!
        clip_actions=False,
        # framework="tf2",
        num_cpus_for_driver=1,
        num_cpus_per_worker=0.5,
        num_gpus=0,
    )

    train(
        SACPIDTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=args.num_seeds,
        num_seeds=1,
        test_mode=True,
        local_mode=True
    )
