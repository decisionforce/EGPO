from drivingforce.expert_in_the_loop.egpo.sac_pid_saver import SACPIDSaverTrainer
from drivingforce.expert_in_the_loop.expert_guided_env import ExpertGuidedEnv
from drivingforce.expert_in_the_loop.common import SaverCallbacks
from drivingforce.train import train, get_train_parser

if __name__ == '__main__':
    args = get_train_parser().parse_args()

    exp_name = "SAC_saver" or args.exp_name
    stop = {"timesteps_total": 2_00_0000}

    config = dict(
        env=ExpertGuidedEnv,
        env_config=dict(
            vehicle_config=dict(
                use_saver=True,
                # 0.8 seems best
                free_level=0.8),
            safe_rl_env=True),
        # ===== Training =====
        takeover_data_discard=True,
        recent_episode_num=3,
        normalize=True,
        only_evaluate_cost=False,
        twin_cost_q=True,
        k_i=1,
        k_p=3,
        k_d=1,
        cost_limit=2,
        optimization=dict(actor_learning_rate=1e-4, critic_learning_rate=1e-4, entropy_learning_rate=1e-4),
        prioritized_replay=False,
        horizon=1500,
        target_network_update_freq=1,
        timesteps_per_iteration=100,
        learning_starts=500,
        clip_actions=False,
        normalize_actions=True,
        num_cpus_for_driver=0.5,
        # No extra worker used for learning. But this config impact the evaluation workers.
        num_cpus_per_worker=0.1,
        # num_gpus_per_worker=0.1 if args.num_gpus != 0 else 0,
        num_gpus=0.2 if args.num_gpus != 0 else 0,

    )

    train(
        SACPIDSaverTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=1,
        num_seeds=1,
        custom_callback=SaverCallbacks,
        test_mode=True,
        local_mode=True
    )
