from __future__ import print_function
import time
from egpo_utils.dagger.exp_saver import Experiment

import os.path as osp

from egpo_utils.process.vis_model_utils import expert_action_prob
from egpo_utils.safe_generalization.run import load_weights
from egpo_utils.expert_guided_env import ExpertGuidedEnv
from egpo_utils import *
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from model import Model
import os
from egpo_utils.common import evaluation_config

# require loguru imageio easydict tensorboardX pyyaml pytorch==1.5.0 stable_baselines3, cudatoolkit==9.2

# hyperpara
NUM_ITS = 5
learning_rate = 5e-4
batch_size = 64
beta_i = 0.3  # expert mix ratio
T = 5000  # batch
evaluation_episode_num = 30
num_epoch = 2000  # sgd epoch on data set
train_loss_threshold = 0.5
device = "cuda"

# training env_config
training_config = dict(
    vehicle_config=dict(
        use_saver=False,
        free_level=100),
    safe_rl_env=True,
    auto_termination=True,
)

# test env config
eval_config = evaluation_config["env_config"]
eval_config["auto_termination"] = True


def make_env(env_cls, config, seed=0):
    def _init():
        env = env_cls(config)
        return env

    return _init


expert_weights = None


def get_expert_action(obs):
    global expert_weights
    if expert_weights is None:
        expert_weights = load_weights(osp.join(osp.dirname(osp.dirname(__file__)), "5_14_safe_expert.npz"))
    saver_a, *_ = expert_action_prob([0, 0], obs, expert_weights, deterministic=False)
    return saver_a


if __name__ == "__main__":
    if not os.path.exists("dagger_models"):
        os.mkdir("dagger_models")
    tm = time.localtime(time.time())
    tm_stamp = "%s-%s-%s-%s-%s-%s" % (tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)
    log_dir = os.path.join(
        "dagger_lr_{}_bs_{}_sgd_iter_{}_dagger_batch_size_{}".format(learning_rate, batch_size, num_epoch, T), tm_stamp)
    exp_log = Experiment()
    exp_log.init(log_dir=log_dir)

    training_env = SubprocVecEnv([make_env(ExpertGuidedEnv, config=training_config)])  # seperate with eval env
    eval_env = ExpertGuidedEnv(eval_config)

    obs_shape = eval_env.observation_space.shape[0]
    action_shape = eval_env.action_space.shape[0]

    # agent
    agent = Model(obs_shape, action_shape, (256, 256)).to(device).float()
    agent.save("dagger_models/model_0.pth")
    model_number = 0
    old_model_number = 0

    # dagger buffer
    samples = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal": [],
    }
    for iteration in range(NUM_ITS):
        steps = 0
        agent = Model(obs_shape, action_shape, (256, 256)).to(device).float()
        agent.load("dagger_models/model_{}.pth".format(model_number))
        curr_beta = beta_i ** model_number

        if model_number != old_model_number:
            old_model_number = model_number

        episode_reward = 0
        success_num = 0
        episode_cost = 0
        done_num = 0
        state = training_env.reset()[0]
        sample_start = time.time()

        while True:
            # preprocess image and find prediction ->  policy(state)
            prediction = agent(torch.tensor(state).to(device).float())
            expert_a = get_expert_action(state)
            pi = curr_beta * expert_a + (1 - curr_beta) * prediction.detach().cpu().numpy().flatten()

            next_state, r, done, info = training_env.step([pi])
            next_state = next_state[0]
            r = r[0]
            done = done[0]
            info = info[0]

            episode_reward += r
            episode_cost += info["native_cost"]
            samples["state"].append(state)
            samples["action"].append(np.array(expert_a))
            samples["next_state"].append(next_state)
            samples["reward"].append(r)
            samples["terminal"].append(done)

            state = next_state
            steps += 1

            # train after T steps
            if steps > T and done:
                if info["arrive_dest"]:
                    success_num += 1
                done_num += 1
                exp_log.scalar(is_train=True, mean_episode_reward=episode_reward / done_num,
                               mean_episode_cost=episode_cost / done_num,
                               success_rate=success_num / done_num,
                               mean_step_reward=episode_reward / steps,
                               beta=curr_beta,
                               sample_time=time.time() - sample_start)
                train_start = time.time()
                store_data(samples, "./data")
                X_train, y_train = read_data("./data", "data_dagger.pkl.gzip")
                loss, last_epoch_loss, epoch_num = train_model(agent, X_train, y_train,
                                                               "dagger_models/model_{}.pth".format(model_number + 1),
                                                               num_epochs=num_epoch * (model_number + 1),
                                                               batch_size=batch_size,
                                                               learning_rate=learning_rate,
                                                               early_terminate_loss_threshold=train_loss_threshold,
                                                               device=device)
                exp_log.scalar(is_train=True,
                               mean_loss=loss,
                               data_set_size=len(X_train),
                               last_epoch_loss=last_epoch_loss,
                               epoch_num=epoch_num,
                               training_time=time.time() - train_start)
                eval_res = evaluation(eval_env, agent, evaluation_episode_num=evaluation_episode_num)
                exp_log.scalar(is_train=False, **eval_res)
                model_number += 1
                break
            if done:
                if info["arrive_dest"]:
                    success_num += 1
                done_num += 1
                training_env.reset()
        exp_log.end_epoch(iteration)
    training_env.close()
    eval_env.close()
