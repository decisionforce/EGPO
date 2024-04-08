"""
Evaluate the trained EGPO agent without the expert.
Please change the `CKPT_PATH` to the path of the checkpoint you want to evaluate.
"""
import pathlib
import tqdm

from egpo_utils.egpo.egpo import EGPOTrainer
from egpo_utils.human_in_the_loop_env import HumanInTheLoopEnv
from egpo_utils.train.utils import initialize_ray

TRAINING_SCRIPT_FOLDER = pathlib.Path(__file__).parent

initialize_ray(test_mode=False)


def get_function(ckpt):
    trainer = EGPOTrainer(dict(

        env=HumanInTheLoopEnv,

        # ===== Training =====
        takeover_data_discard=False,
        alpha=10.0,
        recent_episode_num=5,
        normalize=True,
        twin_cost_q=True,
        k_i=0.01,
        k_p=5,
        # search > 0
        k_d=0.1,

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

    ))

    print("Restoring from checkpoint: ", ckpt)
    trainer.restore(str(ckpt))

    def _f(obs):
        ret = trainer.compute_actions({"default_policy": obs})
        return ret

    return _f


if __name__ == '__main__':

    EPISODE_NUM = 50
    CKPT_PATH = TRAINING_SCRIPT_FOLDER / "EGPO/EGPOTrainer_ExpertGuidedEnv_23216_00000_0_seed=0_2024-04-08_11-59-33/checkpoint_1/checkpoint-1"


    def make_env(env_id=None):
        return HumanInTheLoopEnv(dict(manual_control=False, use_render=False))


    data = []

    env = make_env()

    compute_actions = get_function(CKPT_PATH)

    o = env.reset()
    epi_num = 0

    total_cost = 0
    total_reward = 0
    success_rate = 0
    ep_cost = 0
    ep_reward = 0
    success_flag = False
    horizon = 2000
    step = 0
    with tqdm.tqdm(total=EPISODE_NUM, desc="Episode") as pbar:
        while True:
            # action_to_send = compute_actions(w, [o], deterministic=False)[0]
            step += 1
            action_to_send = compute_actions(o)["default_policy"]
            o, r, d, info = env.step(action_to_send)
            total_reward += r
            ep_reward += r
            total_cost += info["cost"]
            ep_cost += info["cost"]
            if d or step > horizon:
                if info["arrive_dest"]:
                    success_rate += 1
                    success_flag = True
                epi_num += 1
                pbar.update(1)
                if epi_num > EPISODE_NUM:
                    break
                else:
                    o = env.reset()

                data.append({"reward": ep_reward, "success": success_flag, "cost": ep_cost})

                ep_cost = 0.0
                ep_reward = 0.0
                success_flag = False
                step = 0

    print(
        "success_rate:{}, mean_episode_reward:{}, mean_episode_cost:{}".format(success_rate / EPISODE_NUM,
                                                                               total_reward / EPISODE_NUM,
                                                                               total_cost / EPISODE_NUM))

    del compute_actions

    env.close()

    import pandas as pd

    pd.DataFrame(data).to_csv("egpo_eval.csv")
