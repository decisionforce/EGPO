from egpo_utils.egpo.egpo import EGPOTrainer
from egpo_utils.human_in_the_loop_env import HumanInTheLoopEnv

from egpo_utils.train.utils import initialize_ray

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

    trainer.restore(ckpt)

    def _f(obs):
        ret = trainer.compute_actions({"default_policy": obs})
        return ret

    return _f


if __name__ == '__main__':
    def make_env(env_id=None):
        return HumanInTheLoopEnv(dict(manual_control=False, use_render=False))


    from collections import defaultdict

    super_data = defaultdict(list)
    EPISODE_NUM = 50

    env = make_env()
    for ckpt_idx in range(12, 163, 10):
        ckpt = ckpt_idx

        compute_actions = get_function(
            "/home/liquanyi/corl_human_exp/EGPO/SACPIDSaverTrainer_HumanInTheLoopEnv_0689e_00000_0_seed=0_2021-08-24_20-01-33/checkpoint_{}/checkpoint-{}".format(
                ckpt, ckpt)
        )

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
                if epi_num > EPISODE_NUM:
                    break
                else:
                    o = env.reset()

                super_data[ckpt].append({"reward": ep_reward, "success": success_flag, "cost": ep_cost})

                ep_cost = 0.0
                ep_reward = 0.0
                success_flag = False
                step = 0

        print(
            "CKPT:{} | success_rate:{}, mean_episode_reward:{}, mean_episode_cost:{}".format(ckpt,
                                                                                             success_rate / EPISODE_NUM,
                                                                                             total_reward / EPISODE_NUM,
                                                                                             total_cost / EPISODE_NUM))

        del compute_actions

    env.close()

    import json

    try:
        with open("super_data_12_162_10.json", "w") as f:
            json.dump(super_data, f)
    except:
        pass

    print(super_data)
