import os.path as osp
import time
from metadrive.utils.random_utils import get_np_random
from panda3d.core import PNMImage
import gym
import numpy as np
from metadrive.utils.config import Config

from egpo_utils.process.vis_model_utils import expert_action_prob
from egpo_utils.safe_generalization.run import load_weights
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv


class ExpertGuidedEnv(SafeMetaDriveEnv):

    def default_config(self) -> Config:
        """
        Train/Test set both contain 10 maps
        :return: PGConfig
        """
        config = super(ExpertGuidedEnv, self).default_config()
        config.update(dict(
            environment_num=100,
            start_seed=100,
            safe_rl_env_v2=False,  # If True, then DO NOT done even out of the road!
            # _disable_detector_mask=True,  # default False to acc Lidar detection

            # traffic setting
            random_traffic=False,
            traffic_density=0.2,
            traffic_mode="trigger",

            # special setting
            rule_takeover=False,
            takeover_cost=1,
            cost_info="native",  # or takeover
            random_spawn=False,  # used to collect dataset
            cost_to_reward=True,  # for egpo, it accesses the ENV reward by penalty

            vehicle_config=dict(  # saver config, free_level:0 = expert
                use_saver=False,
                free_level=100,
                expert_deterministic=False,
                release_threshold=100,  # the save will be released when level < this threshold
                overtake_stat=False),  # set to True only when evaluate

            expert_value_weights=osp.join(osp.dirname(__file__), "5_14_safe_expert.npz")
        ), allow_overwrite=True)
        return config

    def __init__(self, config):
        # if ("safe_rl_env" in config) and (not config["safe_rl_env"]):
        #     raise ValueError("You should always set safe_rl_env to True!")
        # config["safe_rl_env"] = True
        if config.get("safe_rl_env_v2", False):
            config["out_of_road_penalty"] = 0
        super(ExpertGuidedEnv, self).__init__(config)
        assert self.config["expert_value_weights"] is not None
        self.total_takeover_cost = 0
        self.total_native_cost = 0
        self.state_value = 0
        if self.config["vehicle_config"]["use_saver"]:
            self.expert_weights = load_weights(self.config["expert_value_weights"])
        if self.config["cost_to_reward"]:
            self.config["out_of_road_penalty"] = self.config["out_of_road_cost"]
            self.config["crash_vehicle_penalty"] = self.config["crash_vehicle_cost"]
            self.config["crash_object_penalty"] = self.config["crash_object_cost"]

    def _get_reset_return(self):
        assert self.num_agents == 1
        self.total_takeover_cost = 0
        self.total_native_cost = 0
        if self.config["vehicle_config"]["free_level"] < 1e-3:
            # 1.0 full takeover
            self.vehicle.takeover_start = True
        return super(ExpertGuidedEnv, self)._get_reset_return()

    def step(self, actions):
        obs, r, d, info, = super(ExpertGuidedEnv, self).step(actions)
        info = self.extra_step_info(info)
        return obs, r, d, info

    def extra_step_info(self, step_info):
        # step_info = step_infos[self.DEFAULT_AGENT]

        step_info["native_cost"] = step_info["cost"]
        # if step_info["out_of_road"] and not step_info["arrive_dest"]:
        # out of road will be done now
        step_info["takeover_cost"] = self.config["takeover_cost"] if step_info["takeover_start"] else 0
        self.total_takeover_cost += step_info["takeover_cost"]
        self.total_native_cost += step_info["native_cost"]
        step_info["total_takeover_cost"] = self.total_takeover_cost
        step_info["total_native_cost"] = self.total_native_cost

        if self.config["cost_info"] == "native":
            step_info["cost"] = step_info["native_cost"]
            step_info["total_cost"] = self.total_native_cost
        elif self.config["cost_info"] == "takeover":
            step_info["cost"] = step_info["takeover_cost"]
            step_info["total_cost"] = self.total_takeover_cost
        else:
            raise ValueError
        return step_info

    def _reset_agents(self):
        if self.config["random_spawn"]:
            spawn_lane_index = (*self.vehicle.vehicle_config["spawn_lane_index"][:-1],
                                get_np_random().randint(0, self.current_map.config["lane_num"]))
            self.vehicle.vehicle_config["spawn_lane_index"] = spawn_lane_index
            self.vehicle.vehicle_config["spawn_lateral"] = (get_np_random().rand() - 0.5) * self.current_map.config[
                "lane_width"] / 2

        super(ExpertGuidedEnv, self)._reset_agents()

    def done_function(self, v_id):
        """This function is a little bit different compared to the Env in PGDrive!"""
        done, done_info = super(ExpertGuidedEnv, self).done_function(v_id)
        if self.config["safe_rl_env_v2"]:
            assert self.config["out_of_road_cost"] > 0
            if done_info["out_of_road"]:
                done = False
        return done, done_info

    def saver(self, v_id: str, actions):
        """
        Action prob takeover
        """
        if self.config["rule_takeover"]:
            return self.rule_takeover(v_id, actions)
        vehicle = self.vehicles[v_id]
        action = actions[v_id]
        steering = action[0]
        throttle = action[1]
        self.state_value = 0
        pre_save = vehicle.takeover
        if vehicle.vehicle_config["use_saver"] or vehicle._expert_takeover:
            # saver can be used for human or another AI
            free_level = vehicle.vehicle_config["free_level"] if not vehicle._expert_takeover else 1.0
            obs = self.observations[v_id].observe(vehicle)
            try:
                saver_a, a_0_p, a_1_p = expert_action_prob(action, obs, self.expert_weights,
                                                           deterministic=vehicle.vehicle_config["expert_deterministic"])
            except ValueError:
                print("Expert can not takeover, due to observation space mismathing!")
                saver_a = action
            else:
                if free_level <= 1e-3:
                    steering = saver_a[0]
                    throttle = saver_a[1]
                elif free_level > 1e-3:
                    if a_0_p * a_1_p < 1 - vehicle.vehicle_config["free_level"]:
                        steering, throttle = saver_a[0], saver_a[1]

        # indicate if current frame is takeover step
        vehicle.takeover = True if action[0] != steering or action[1] != throttle else False
        saver_info = {
            "takeover_start": True if not pre_save and vehicle.takeover else False,
            "takeover_end": True if pre_save and not vehicle.takeover else False,
            "takeover": vehicle.takeover if pre_save else False
        }
        if saver_info["takeover"]:
            saver_info["raw_action"] = [steering, throttle]
        return (steering, throttle) if saver_info["takeover"] else action, saver_info

    def rule_takeover(self, v_id, actions):
        vehicle = self.vehicles[v_id]
        action = actions[v_id]
        steering = action[0]
        throttle = action[1]
        if vehicle.vehicle_config["use_saver"] or vehicle._expert_takeover:
            # saver can be used for human or another AI
            save_level = vehicle.vehicle_config["save_level"] if not vehicle._expert_takeover else 1.0
            obs = self.observations[v_id].observe(vehicle)
            try:
                saver_a, a_0_p, a_1_p = expert_action_prob(action, obs, self.expert_weights,
                                                           deterministic=vehicle.vehicle_config["expert_deterministic"])
            except ValueError:
                print("Expert can not takeover, due to observation space mismathing!")
            else:
                if save_level > 0.9:
                    steering = saver_a[0]
                    throttle = saver_a[1]
                elif save_level > 1e-3:
                    heading_diff = vehicle.heading_diff(vehicle.lane) - 0.5
                    f = min(1 + abs(heading_diff) * vehicle.speed * vehicle.max_speed, save_level * 10)
                    # for out of road
                    if (obs[0] < 0.04 * f and heading_diff < 0) or (obs[1] < 0.04 * f and heading_diff > 0) or obs[
                        0] <= 1e-3 or \
                            obs[
                                1] <= 1e-3:
                        steering = saver_a[0]
                        throttle = saver_a[1]
                        if vehicle.speed < 5:
                            throttle = 0.5
                    # if saver_a[1] * vehicle.speed < -40 and action[1] > 0:
                    #     throttle = saver_a[1]

                    # for collision
                    lidar_p = vehicle.lidar.get_cloud_points()
                    left = int(vehicle.lidar.num_lasers / 4)
                    right = int(vehicle.lidar.num_lasers / 4 * 3)
                    if min(lidar_p[left - 4:left + 6]) < (save_level + 0.1) / 10 or min(lidar_p[right - 4:right + 6]
                                                                                        ) < (save_level + 0.1) / 10:
                        # lateral safe distance 2.0m
                        steering = saver_a[0]
                    if action[1] >= 0 and saver_a[1] <= 0 and min(min(lidar_p[0:10]), min(lidar_p[-10:])) < save_level:
                        # longitude safe distance 15 m
                        throttle = saver_a[1]

        # indicate if current frame is takeover step
        pre_save = vehicle.takeover
        vehicle.takeover = True if action[0] != steering or action[1] != throttle else False
        saver_info = {
            "takeover_start": True if not pre_save and vehicle.takeover else False,
            "takeover_end": True if pre_save and not vehicle.takeover else False,
            "takeover": vehicle.takeover if pre_save else False
        }
        return (steering, throttle) if saver_info["takeover"] else action, saver_info

    def capture(self):

        img = PNMImage()
        self.pg_world.win.getScreenshot(img)
        img.write("main_{}.jpg".format(time.thread_time()))

    def reward_function(self, vehicle_id: str):
        ret = super(ExpertGuidedEnv, self).reward_function(vehicle_id)
        return ret


class SmartExpertGuidedEnv(ExpertGuidedEnv):

    def saver(self, v_id: str, tuple_action):
        assert len(tuple_action[v_id]) == 3, "action space error"
        vehicle = self.vehicles[v_id]
        vehicle.vehicle_config["free_level"] = tuple_action[v_id][-1]
        return super(SmartExpertGuidedEnv, self).saver(v_id, tuple_action)

    @property
    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)


if __name__ == '__main__':
    e = SmartExpertGuidedEnv(dict(use_render=True, vehicle_config=dict(use_saver=True)))
    print(e.action_space)
    print(e.observation_space)
    e.reset()
    for _ in range(100000):
        a = e.action_space.sample()
        a[-1] = 1
        i = e.step(a)
        e.render(text=dict(action=a))
    e.close()
