from egpo_utils.safe_generalization.callback import SafeGeneralizationCallbacks
from egpo_utils.safety.ppo_lag import PPOLag
from egpo_utils.human_in_the_loop_env import HumanInTheLoopEnv
from egpo_utils.train import train, get_train_parser
import datetime

def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


if __name__ == '__main__':
    args = get_train_parser().parse_args()

    exp_name = "PPO_{}".format(get_time_str()) or args.exp_name
    stop = {"timesteps_total": 2_00000}

    config = dict(
        env=HumanInTheLoopEnv,
        env_config={"manual_control":False},

        # ===== Training =====
        horizon=400,
        num_sgd_iter=20,
        lr=5e-5,
        grad_clip=10.0,
        rollout_fragment_length=200,
        sgd_minibatch_size=100,
        train_batch_size=2000,
        num_gpus=0.5 if args.num_gpus != 0 else 0,
        num_cpus_per_worker=0.1,
        num_cpus_for_driver=0.5,
        num_workers=5,
        clip_actions=False
    )

    train(
        PPOLag,
        exp_name=exp_name,
        keep_checkpoints_num=None,
        checkpoint_freq=1,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=2,
        num_seeds=1,
        custom_callback=SafeGeneralizationCallbacks,
        # test_mode=True,
        # local_mode=True
    )
