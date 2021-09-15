from pgdrive.envs.generation_envs.safe_pgdrive_env import SafePGDriveEnv
from ray import tune

from drivingforce.safety.sac_pid.sac_pid import SACPIDTrainer
from drivingforce.safe_generalization.callback import SafeGeneralizationCallbacks
from drivingforce.train import train, get_train_parser


class Env(SafePGDriveEnv):
    def __init__(self, config):
        config["speed_reward"] = 1.
        config["out_of_road_cost"] = 1.
        super(SafePGDriveEnv, self).__init__(config)
        self.total_takeover_cost = 0

    def _get_reset_return(self):
        self.total_takeover_cost = 0
        return super(SafePGDriveEnv, self)._get_reset_return()

    def step(self, actions):
        obs, r, d, info, = super(SafePGDriveEnv, self).step(actions)
        self.total_takeover_cost += info["cost"]
        info["total_cost"] = self.total_takeover_cost
        return obs, r, d, info

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        current_lane = vehicle.lane if vehicle.lane in vehicle.routing_localization.current_ref_lanes else \
            vehicle.routing_localization.current_ref_lanes[0]
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        lateral_factor = 1.0
        current_road = vehicle.current_road
        positive_road = 1 if not current_road.is_negative_road() else -1

        reward = 0.0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
        reward += self.config["speed_reward"] * (vehicle.speed / vehicle.max_speed) * positive_road

        step_info["step_reward"] = reward

        if vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        elif vehicle.out_of_route:
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.arrive_destination:
            reward = +self.config["success_reward"]
        return reward, step_info


if __name__ == '__main__':
    args = get_train_parser().parse_args()

    exp_name = args.exp_name or "TEST"
    stop = int(1000000)

    config = dict(
        env=Env,
        env_config=dict(environment_num=100, start_seed=99),

        # ===== Evaluation =====
        evaluation_interval=3,
        evaluation_num_episodes=50,
        evaluation_config=dict(env_config=dict(environment_num=50, start_seed=0)),
        evaluation_num_workers=2,
        metrics_smoothing_episodes=50,

        # ===== Training =====
        # Best:
        # 0.008 learning rate
        # recent episode num: 3
        recent_episode_num=3,
        normalize=True,
        only_evaluate_cost=False,
        twin_cost_q=True,
        k_i=tune.grid_search([0.001, 0]),
        k_p=tune.grid_search([16,8,4,2,1]),
        cost_limit=2,
        optimization=dict(actor_learning_rate=1e-4, critic_learning_rate=1e-4, entropy_learning_rate=1e-4),
        prioritized_replay=False,
        horizon=1500,
        target_network_update_freq=1,
        timesteps_per_iteration=1000,
        learning_starts=10000,
        clip_actions=False,
        normalize_actions=True,
        num_cpus_for_driver=1,
        # No extra worker used for learning. But this config impact the evaluation workers.
        num_cpus_per_worker=0.1,
        # num_gpus_per_worker=0.1 if args.num_gpus != 0 else 0,
        num_gpus=0.2 if args.num_gpus != 0 else 0,
    )

    train(
        SACPIDTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=args.num_seeds,
        num_seeds=3,
        custom_callback=SafeGeneralizationCallbacks,
        # num_seeds=1,
        # test_mode=True,
        # local_mode=True
    )
