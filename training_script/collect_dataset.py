import json
from ray.rllib.policy.sample_batch import SampleBatch

from egpo_utils.expert_guided_env import ExpertGuidedEnv
from egpo_utils.common import get_expert_action


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
    num = int(500)
    pool = []

    env = ExpertGuidedEnv(dict(vehicle_config=dict(free_level=1.0, use_saver=True), use_render=False))
    success = 0

    obs = env.reset()

    episode_num = 0
    last = 0
    while episode_num < num:
        last += 1
        action = get_expert_action(env)
        new_obs, reward, done, info = env.step(action)
        pool.append({SampleBatch.OBS: obs.tolist(), SampleBatch.ACTIONS: action.tolist(), SampleBatch.NEXT_OBS: new_obs.tolist(),
                     SampleBatch.DONES: done,
                     SampleBatch.REWARDS: reward, SampleBatch.INFOS: process_info(info)})
        obs = new_obs
        if done:
            episode_num += 1
            if info["arrive_dest"]:
                success += 1
            print('reset:', episode_num, "this_episode_len:", last, "total_success_rate:", success / episode_num)
            obs = env.reset()
            last = 0
            print('finish {}'.format(episode_num))
    with open('expert_traj_' + str(num) + '.json', 'w') as f:
        json.dump(pool, f)
