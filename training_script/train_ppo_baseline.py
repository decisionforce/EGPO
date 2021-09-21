from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray import tune
from egpo_utils.expert_guided_env import ExpertGuidedEnv
from egpo_utils.common import EGPOCallbacks, evaluation_config
from egpo_utils.train import train, get_train_parser

if __name__ == '__main__':
    args = get_train_parser().parse_args()

    exp_name = "PPO_native"
    stop = {"timesteps_total": 20_000_000}

    config = dict(
        env=ExpertGuidedEnv,
        env_config=dict(
            crash_vehicle_penalty=1.,
            crash_object_penalty=0.5,
            out_of_road_penalty=1.,
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
        metrics_smoothing_episodes=20,

        # ===== Training =====
        horizon=1500,
        num_sgd_iter=20,
        lr=5e-5,
        grad_clip=10.0,
        rollout_fragment_length=200,
        sgd_minibatch_size=100,
        train_batch_size=4000,
        num_gpus=0.5 if args.num_gpus != 0 else 0,
        num_cpus_per_worker=0.1,
        num_cpus_for_driver=0.5,
        num_workers=8,
        clip_actions=False
    )

    train(
        PPOTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=None,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        num_seeds=5,
        # num_seeds=1,
        custom_callback=EGPOCallbacks,
        # test_mode=True,
        # local_mode=True
    )
