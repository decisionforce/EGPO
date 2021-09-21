import json

import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch

from egpo_utils.human_in_the_loop_env import HumanInTheLoopEnv
from egpo_utils.common import expert_action_prob


def process_info(info):
    ret = {}
    for k, v in info.items():
        # filter float 32
        if k != "raw_action":
            ret[k] = v
    return ret


if __name__ == '__main__':
    """
    Data = Tuple[o, a, d, r, i]
    """
    num = int(20)
    pool = []

    env = HumanInTheLoopEnv(dict(manual_control=True, use_render=True))
    success = 0
    episode_reward = []
    episode_cost = []

    total_reward = 0
    total_cost = 0

    obs = env.reset()

    episode_num = 0
    last = 0
    while episode_num < num:
        last += 1
        new_obs, reward, done, info = env.step([0, 0])
        action = info["raw_action"]
        total_cost += info["cost"]
        pool.append({SampleBatch.OBS: list(obs), SampleBatch.ACTIONS: list(action), SampleBatch.NEXT_OBS: list(new_obs),
                     SampleBatch.DONES: done,
                     SampleBatch.REWARDS: reward, SampleBatch.INFOS: process_info(info)})
        obs = new_obs
        total_reward += reward
        if done:
            episode_num += 1
            if info["arrive_dest"]:
                success += 1
            episode_reward.append(total_reward)
            episode_cost.append(total_cost)
            total_reward = 0
            total_cost = 0
            print('reset:', episode_num, "this_episode_len:", last, "total_success_rate:", success / episode_num,
                  "mean_episode_reward:{}({})".format(np.mean(episode_reward),np.std(episode_reward)),
                  "mean_episode_cost:{}({})".format(np.mean(episode_cost), np.std(episode_cost)))
            obs = env.reset()
            last = 0
            print('finish {}'.format(episode_num))
    with open('expert_traj_' + str(num) + '.json', 'w') as f:
        json.dump(pool, f)
