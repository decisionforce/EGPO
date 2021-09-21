from egpo_utils.ppo_lag.ppo_lag import PPOLag
from ray import tune
from egpo_utils.common import EGPOCallbacks, evaluation_config
from egpo_utils.expert_guided_env import ExpertGuidedEnv
from egpo_utils.train import train, get_train_parser

if __name__ == '__main__':
    args = get_train_parser().parse_args()

    exp_name = args.exp_name or "PPO_LAG_native"
    stop = int(100_0000)

    config = dict(
        env=ExpertGuidedEnv,
        env_config=dict(
            vehicle_config=dict(
                use_saver=False,
                free_level=100),
            safe_rl_env=True,
        ),

        # ===== Evaluation =====
        evaluation_interval=1,
        evaluation_num_episodes=30,
        evaluation_config=evaluation_config,
        evaluation_num_workers=2,
        metrics_smoothing_episodes=50,

        # ===== Training =====
        # Best:
        # recent episode num: 3
        cost_limit=1,
        horizon=1500,
        # num_sgd_iter=10,
        lr=5e-5,
        rollout_fragment_length=200,
        # sgd_minibatch_size=100,
        train_batch_size=4000,
        num_gpus=0.5 if args.num_gpus != 0 else 0,
        num_cpus_per_worker=0.2,
        num_cpus_for_driver=1,
        num_workers=4,
        clip_actions=False
    )

    train(
        PPOLag,
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=args.num_seeds,
        num_seeds=5,
        custom_callback=EGPOCallbacks,
    )
