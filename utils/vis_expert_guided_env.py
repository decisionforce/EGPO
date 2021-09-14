from drivingforce.expert_in_the_loop.expert_guided_env import ExpertGuidedEnv
from pgdrive.world.onscreen_message import PGOnScreenMessage

if __name__ == "__main__":
    PGOnScreenMessage.SCALE = 0.1
    env = ExpertGuidedEnv(dict(
        vehicle_config=dict(
            use_saver=False,
            free_level=0.95,
            overtake_stat=True,
            expert_deterministic=False,
            # spawn_lateral=7,
            # increment_steering=True
            show_navi_mark=False
          ),
        # camera_height=7,
        # accident_prob=1.0,
        # traffic_density=0.3,
        # traffic_mode="respawn",
        cull_scene=True,
        map_config={
            "config": 5,
        },
        # camera_dist=10,
        pg_world_config={"show_fps":False},
        cost_to_reward=True,
        # crash_vehicle_penalty=1.,
        # crash_object_penalty=0.5,
        # out_of_road_penalty=1.,
        # crash_object_penalty=5,
        # crash_vehicle_penalty=5,
        # rule_takeover=True,

        safe_rl_env=True,
        start_seed=108,
        environment_num=1,

        use_render=True,
        debug=True,
        manual_control=True))

    def _save(env):
        env.vehicle.vehicle_config["use_saver"]= not env.vehicle.vehicle_config["use_saver"]

    eval_reward = []
    done_num=0
    o = env.reset()
    # env.vehicle.remove_display_region()
    env.main_camera.set_follow_lane(True)
    env.pg_world.accept("p",env.capture)
    env.pg_world.accept("u", _save, extraArgs=[env])
    max_s = 0
    max_t = 0
    start = 0
    total_r = 0
    for i in range(1, 30000):
        o_to_evaluate = o
        o, r, d, info = env.step(env.action_space.sample())
        total_r += r
        max_s = max(max_s, info["raw_action"][0])
        max_t = max(max_t, info["raw_action"][1])

        # assert not info["takeover_start"]
        text = {
                # "save": env.vehicle.takeover, "overtake_num": info["overtake_vehicle_num"],
                # "native_cost": info["native_cost"], "total_native_cost": info["total_native_cost"],
                "reward": total_r, "takeover_cost": info["takeover_cost"],
                # "total_takeover_cost": info["total_takeover_cost"],
                # "takeover start": info["takeover_start"], "takeover end": info["takeover_end"],
                "Takeover": info["takeover"],
                # "raw_action": env.vehicle.last_current_action[1],
                # "state_value": env.state_value,
                # "map_seed": env.current_map.random_seed,
                "Cost":int(info["total_native_cost"]),
                # "total_cost":info["total_cost"],
                # "crash_vehicle":info["crash_vehicle"],
                # "crash_object":info["crash_object"]
                # "current_map":env.current_map.random_seed
        }
        if env.config["cost_info"] == "native":
            assert info["cost"] == info["native_cost"]
            assert info["total_cost"] == info["total_native_cost"]
        elif env.config["cost_info"] == "takeover":
            assert info["cost"] == info["takeover_cost"]
            assert info["total_cost"] == info["total_takeover_cost"]
        else:
            raise ValueError
        # if info["takeover_start"] and not env.config["manual_control"]:
        #     print(info["raw_action"])
        #     assert info["raw_action"] == (0,1)
        #     assert not info["takeover"]
        # if info["takeover"] and not env.config["manual_control"]:
        #     print(info["raw_action"])
        #     assert info["raw_action"] != (0,1)
        # print(r)
        env.render(text=text)
        if d:
            eval_reward.append(total_r)
            done_num+=1
            if done_num > 100:
                break
            print(info["out_of_road"])
            print("done_cost:{}".format(info["cost"]))
            print("done_reward:{}".format(r))
            print("total_takeover_cost:{}".format(info["total_takeover_cost"]))
            takeover_cost = 0
            native_cost = 0
            total_r = 0
            print("episode_len:", i - start)
            env.reset()
            start = i
    import numpy as np
    print(np.mean(eval_reward),np.std(sorted(eval_reward)))
    env.close()
