from egpo_utils.process.restore import restore
import copy
from egpo_utils.train.utils import initialize_ray
from egpo_utils.generalization.rgb.network import register_our_network
from egpo_utils.egpo.sac_pid_saver import SACPIDSaverTrainer
from metadrive.world.onscreen_message import PGOnScreenMessage

if __name__ == "__main__":
    PGOnScreenMessage.SCALE = 0.1
    initialize_ray(test_mode=True)
    register_our_network()
    ckpt = "F:\\CoRL_NIPS_CVPR\\plot\\saver_sac_pid\\SACPIDSaverTrainer_SafeDrivingEnv_20512_00009_9_alpha=3.0,seed=300_2021-05-16_15-27-10\\checkpoint_170\\checkpoint-170"

    # trainer = restore(ckpt, "PPO", dict(evaluation_num_workers=0, evaluation_interval=0, num_gpus=0), True)
    trainer = restore(ckpt, SACPIDSaverTrainer, dict(evaluation_num_workers=0, evaluation_interval=0, num_gpus=0), True)

    import time
    from egpo_utils.expert_guided_env import ExpertGuidedEnv
    from egpo_utils.common import evaluation_config

    eval_config = copy.deepcopy(evaluation_config)
    eval_config["env_config"]["use_render"] = True
    eval_config["env_config"]["vehicle_config"]["use_saver"] = True
    eval_config["env_config"]["vehicle_config"]["free_level"] = 0.95
    eval_config["env_config"]["vehicle_config"]["show_navi_mark"] = False
    eval_config["env_config"]["environment_num"] = 1
    eval_config["env_config"]["start_seed"] = 502

    # eval_config["env_config"]["traffic_density"]=0.35
    # eval_config["env_config"]["traffic_mode"]="respawn"
    # eval_config["env_config"]["start_seed"]=537
    # eval_config["env_config"]["environment_num"]=100
    # 535 515 532 541 531 538 502 547 505 516 548 531

    env = ExpertGuidedEnv(eval_config["env_config"])
    obs = env.reset()
    env.main_camera.set_follow_lane(True)
    start = time.time()
    ep_reward = 0
    ep_length = 0
    for s in range(100000):
        action = trainer.compute_action(obs)
        obs, r, d, i = env.step(action)
        ep_reward += r
        ep_length += 1
        env.render(text={"Cost": int(i["total_native_cost"]),
                         "Takeover": i["takeover"]})
        if d:
            print("This episode reward: {}. Length {}.".format(ep_reward, ep_length))
            ep_reward = 0
            ep_length = 0
            env.reset()
        if (s + 1) % 100 == 0:
            print(f"{s + 1}/{1000} Time Elapse: {time.time() - start}")
    print(f"Total Time Elapse: {time.time() - start}")
