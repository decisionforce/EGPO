from egpo_utils.common import SaverCallbacks
from egpo_utils.egpo.sac_pid_saver import SACPIDSaverTrainer
from egpo_utils.human_in_the_loop_env import HumanInTheLoopEnv
from egpo_utils.train import train, get_train_parser
import datetime

def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


if __name__ == '__main__':
    args = get_train_parser().parse_args()

    exp_name = "EGPO_{}".format(get_time_str()) or args.exp_name
    stop = {"timesteps_total": 20_0000}

    config = dict(
        env=HumanInTheLoopEnv,
        env_config={
            "manual_control": True,
            "use_render": True,
            "window_size":(1600, 1100)
        },


        # ===== Training =====
        takeover_data_discard=False,
        alpha=10.0,
        recent_episode_num=5,
        normalize=True,
        twin_cost_q=True,
        # no_reward=tune.grid_search([True, False]),
        k_i=0.01,
        k_p=5,
        # search > 0
        k_d=0.1,
        # k_i=tune.grid_search([0.01, 0.005, 0.001]),

        # expected max takeover num
        cost_limit=300,
        optimization=dict(actor_learning_rate=1e-4, critic_learning_rate=1e-4, entropy_learning_rate=1e-4),
        prioritized_replay=False,
        horizon=400,
        target_network_update_freq=1,
        timesteps_per_iteration=100,
        metrics_smoothing_episodes=10,
        learning_starts=100,
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
        keep_checkpoints_num=None,
        checkpoint_freq=1,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=2,
        num_seeds=1,
        custom_callback=SaverCallbacks,
        # test_mode=True,
        # local_mode=True
    )
