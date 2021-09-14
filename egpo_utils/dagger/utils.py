from __future__ import print_function

import gzip
import json
import os
import pickle

import numpy as np
import torch


def save_results(episode_rewards, results_dir="./results", result_file_name="training_result"):
    # save results
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # save statistics in a dictionary and write them into a .json file
    results = dict()
    results["number_episodes"] = len(episode_rewards)
    results["episode_rewards"] = episode_rewards

    results["mean_all_episodes"] = np.array(episode_rewards).mean()
    results["std_all_episodes"] = np.array(episode_rewards).std()

    fname = os.path.join(results_dir, result_file_name)
    fh = open(fname, "w")
    json.dump(results, fh)
    print('... finished')


def store_data(data, datasets_dir="./data"):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'data_dagger.pkl.gzip')
    f = gzip.open(data_file, 'wb')
    pickle.dump(data, f)


def read_data(datasets_dir="./data", path='data.pkl.gzip', frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    data_file = os.path.join(datasets_dir, path)

    f = gzip.open(data_file, 'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')
    return X, y


def train_model(model, X_train, y_train, path, num_epochs=50, learning_rate=1e-3, lambda_l2=1e-5,
                early_terminate_loss_threshold=0.2,
                batch_size=32, device="cuda"):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lambda_l2)  # built-in L2
    # shuffle
    perm = np.arange(len(X_train))
    np.random.shuffle(perm)
    X_train = X_train[perm]
    y_train = y_train[perm]

    X_train_torch = torch.from_numpy(X_train).to(device).float()
    y_train_torch = torch.from_numpy(y_train).to(device).float()
    total_loss = []
    sgd_num = len(X_train_torch) / batch_size
    t = 0,
    epoch_loss = 0
    for t in range(num_epochs):
        epoch_loss = 0
        for i in range(0, len(X_train_torch), batch_size):
            curr_X = X_train_torch[i:i + batch_size]
            curr_Y = y_train_torch[i:i + batch_size]
            preds = model(curr_X)
            loss = criterion(preds, curr_Y)
            with torch.no_grad():
                total_loss.append(loss.item())
                epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch_loss / sgd_num < early_terminate_loss_threshold:
            break
    model.save(path)
    return np.sum(total_loss) / len(total_loss), epoch_loss / sgd_num, t


def evaluation(env, model, evaluation_episode_num=30, device="cuda"):
    with torch.no_grad():
        print("... evaluation")
        episode_reward = 0
        episode_cost = 0
        success_num = 0
        episode_num = 0
        velocity = []
        episode_overtake = []
        state = env.reset()
        while episode_num < evaluation_episode_num:
            prediction = model(torch.tensor(state).to(device).float())
            next_state, r, done, info = env.step(prediction.detach().cpu().numpy().flatten())
            state = next_state
            episode_reward += r
            episode_cost += info["native_cost"]
            velocity.append(info["velocity"])
            if done:
                episode_overtake.append(info["overtake_vehicle_num"])
                episode_num += 1
                if info["arrive_dest"]:
                    success_num += 1
                env.reset()
        res = dict(
            mean_episode_reward=episode_reward / episode_num,
            mean_episode_cost=episode_cost / episode_num,
            mean_success_rate=success_num / episode_num,
            mean_velocity=np.mean(velocity),
            mean_episode_overtake_num=np.mean(episode_overtake)
        )
        return res
