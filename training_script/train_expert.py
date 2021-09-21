"""
Expert is trained by this script
"""

from egpo_utils.common import EGPOCallbacks
from egpo_utils.expert_guided_env import ExpertGuidedEnv
from egpo_utils.ppo_lag.ppo_lag import PPOLag
from ray.rllib.agents.ppo.ppo import PPOTrainer
from egpo_utils.train import train, get_train_parser
from egpo_utils.common import evaluation_config
from ray import tune

if __name__ == '__main__':
    args = get_train_parser()
    args.add_argument("--PPO", action="store_false")
    args = args.parse_args()
    trainer = PPOLag if not args.PPO else PPOTrainer

    exp_name = "PPO_lag_expert" if not args.exp_name else args.exp_name
    stop = int(20_0000_0000)

    config = dict(
        env=ExpertGuidedEnv,
        env_config={
            "crash_object_cost": 2,
            "crash_object_penalty": 10,
            "crash_vehicle_cost": 2,
            "crash_vehicle_penalty": 10,
            "out_of_road_cost": 2,
            "out_of_road_penalty": 5,
            "safe_rl_env": False,
            "use_lateral": True,
            "vehicle_config": {"use_saver": False}
        },

        # ===== Evaluation =====
        evaluation_interval=3,
        evaluation_num_episodes=50,
        evaluation_config=evaluation_config,
        evaluation_num_workers=2,
        metrics_smoothing_episodes=50,

        # ===== Training =====
        horizon=1000,
        num_sgd_iter=20,
        lr=5e-5,
        rollout_fragment_length=200,
        sgd_minibatch_size=2048,
        train_batch_size=160000,
        num_gpus=0.2 if args.num_gpus != 0 else 0,
        num_cpus_per_worker=0.1,
        num_cpus_for_driver=1,
        num_workers=10,
    )

    if not args.PPO:
        config["cost_limit"] = 1.0

    train(
        trainer,
        exp_name=exp_name,
        custom_callback=EGPOCallbacks,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        restore="",
        num_gpus=args.num_gpus,
        # num_seeds=3,
        num_seeds=4,
        test_mode=args.test,
        # local_mode=True
    )
