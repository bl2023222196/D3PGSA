import random
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
from sklearn.cluster import DBSCAN
import torch.nn.functional as F
from itertools import islice
import matplotlib.pyplot as plt
import collections
from chargenv5 import Env
from experience_generation_model import main
from feature_generation_model import simsiam
from collections import deque

torch.autograd.set_detect_anomaly(True)


class ReplayBuffer:
    def __init__(self, capacity, eps=1, min_samples=5):
        self.buffer = collections.deque(maxlen=capacity)
        self.clusters = {}
        self.eps = eps
        self.min_samples = min_samples

    def add(self, experience):
        # experience is expected to be a tuple (feature_vector, action, reward, next_feature_vector, done)
        self.buffer.append(experience)

    def cluster(self, features):
        # Assumes the first element of each experience in the buffer is the feature vector for clustering

        if len(features) == 0:
            return
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(features.cpu())
        self.clusters = {}  # Reset clusters
        for i, label in enumerate(labels):
            if label not in self.clusters:
                self.clusters[label] = []
            self.clusters[label].append(self.buffer[i])

    def sample_no_cluster(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return transitions
    def sample(self, batch_size):
        # Ensure there's at least one sample per cluster
        num_clusters = len(self.clusters)
        if num_clusters == 0:
            return []

        samples_per_cluster = max(1, batch_size // num_clusters)
        samples = []

        for cluster in self.clusters.values():
            if len(cluster) < samples_per_cluster:
                samples += cluster
            else:
                samples += random.sample(cluster, samples_per_cluster)

        # If we don't have enough samples due to rounding, sample from the entire buffer
        while len(samples) < batch_size:
            samples.append(random.choice(self.buffer))

        return samples[:batch_size]

    def size(self):
        return len(self.buffer)




class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 256)
        self.fc3 = torch.nn.Linear(256, 64)
        self.fc4 = torch.nn.Linear(64, action_dim)
        self.action_bound = action_bound

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x)) * self.action_bound
        return x


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 256)
        self.fc3 = torch.nn.Linear(256, 64)
        self.fc_out = torch.nn.Linear(64, 1)


    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc_out(x)


class DDPG:
    ''' DDPG算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.lr = actor_lr

        self.actor.load_state_dict(torch.load("./model/actor_initial5.pth"))
        self.critic.load_state_dict(torch.load("./model/critic_initial5.pth"))
        self.critic2.load_state_dict(torch.load("./model/critic2_initial5.pth"))
        # init.normal_(self.actor.weight, mean=0.0, std=0.01)
        # init.zeros_(self.actor.bias)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)



        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, weight_decay=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, weight_decay=0.001)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.lr, weight_decay=0.001)


        self.gamma = gamma
        self.sigma = sigma
        self.tau = tau
        self.action_dim = action_dim
        self.device = device

    def decay_sigma(self):
        self.sigma *= 0.98

    def decay_lr(self):
        self.lr *= 0.5

    def take_action(self, state):
        state = state.to(self.device)
        action = self.actor(state).cpu()

        action = action.detach() + self.sigma * torch.randn(self.action_dim, dtype=torch.float32)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        stacked_tensors = torch.cat((transition_dict), dim=1).squeeze(0).to(self.device)  # 移除中间的维度，得到 64x11
        states = stacked_tensors[:, :4]
        actions = stacked_tensors[:, 4:5]
        rewards = stacked_tensors[:, 5:6]
        next_states = stacked_tensors[:, 6:10]
        dones = stacked_tensors[:, 10:11]
        # next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        # next_q_values = self.normalize(next_q_values)
        target_Q1 = self.target_critic(next_states, self.target_actor(next_states))
        target_Q2 = self.target_critic2(next_states, self.target_actor(next_states))
        target_Q = torch.min(target_Q1, target_Q2)
        q_targets = rewards + self.gamma * target_Q * (1 - dones)
        # q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = F.mse_loss(self.critic(states, actions), q_targets, reduction='none')
        td_error1 = critic_loss.clone()
        critic_loss = (critic_loss).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        critic2_loss = F.mse_loss(self.critic2(states, actions), q_targets, reduction='none')
        td_error2 = critic2_loss.clone()
        critic2_loss = (critic2_loss).mean()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()


        if torch.equal(target_Q, target_Q1):
            actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        else:
            actor_loss = -torch.mean(self.critic2(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
        self.soft_update(self.critic2, self.target_critic2)


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, device):
    return_list = []
    max_reward = 0
    first = True

    for i_episode in range(num_episodes):
        episode_return = 0
        state = env.reset()

        agent.decay_sigma()

        done = False

        while not done:
            action = agent.take_action(state).to(device)
            next_state, reward, done = env.step(torch.transpose(action, 0, 1))
            experience = torch.cat((state, action, reward.view(1, 1), next_state,
                                    torch.tensor(done, dtype=torch.float32, device=device).view(1, 1)), dim=1)
            replay_buffer.add(experience.unsqueeze(1))

            state = next_state


            episode_return += reward.item()
            if replay_buffer.size() > minimal_size:
                if first:
                    if os.path.isfile('./model/cm_model%d.pth' % Num_data):
                        E = main.generate(Num_data)
                        E = torch.cat(E, dim=0).squeeze(1)

                        for i in range(E.size(0)):
                            for j in range(E.size(1)):
                                E[i, j, 3] = j + 1
                                E[i, j, 9] = j + 1
                                if j != E.size(1) - 1:
                                    E[i, j, -1] = 0
                                else:
                                    E[i, j, -1] = 1

                                replay_buffer.add(E[i, j].unsqueeze(0).unsqueeze(0))
                                first = False
                        del E
                        simsiam.train(replay_buffer.buffer, Num_data)
                        model = simsiam.test(Num_data)
                        features = model(torch.cat(list(replay_buffer.buffer), dim=0))[-1]
                        replay_buffer.cluster(features)
                        del model
                    else:
                        e = replay_buffer.sample_no_cluster(minimal_size)
                        num_tensors_to_concatenate = len(e) - (len(e) % 24)
                        data = [torch.cat(e[i:i + 24], dim=1) for i in range(0, num_tensors_to_concatenate, 24)]
                        main.train(data, Num_data)

                        E = main.generate(Num_data)
                        E = torch.cat(E, dim=0).squeeze(1)
                        for i in range(E.size(0)):
                            for j in range(E.size(1)):
                                if j != E.size(1) - 1:
                                    E[i, j, -1] = 0
                                    E[i, j, -1] = 1

                                replay_buffer.add(E[i, j].unsqueeze(0).unsqueeze(0))
                                first = False
                        simsiam.train(replay_buffer.buffer, Num_data)
                        model = simsiam.test(Num_data)
                        features = model(torch.cat(list(replay_buffer.buffer), dim=0))[-1]
                        replay_buffer.cluster(features)
                        del model
                else:
                    e = replay_buffer.sample(batch_size)
                    agent.update(e)

        return_list.append(episode_return)
        if (i_episode + 1) % 10 == 0:
            print(
                'episodes: %d, reward: %f' % (i_episode + 1, torch.mean(torch.tensor(return_list[-10:], dtype=float))))
            if replay_buffer.size() > minimal_size:
                model = simsiam.test(Num_data)
                features = model(torch.cat(list(replay_buffer.buffer), dim=0))[-1]
                replay_buffer.cluster(features)
                del model
    return return_list, max_reward


actor_lr = 0.002
critic_lr = 0.002
num_episodes = 1000

hidden_dim = 128
gamma = 0.98
tau = 0.001
buffer_size = 40000

minimal_size = 5000
batch_size = 128
sigma = 0.5
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

Num_data = 6
dataset = 'EV Charging Reports'
data_path = './datasets/{}.csv'.format(dataset)
env = Env(1, 0, 0, Num_data, data_path)

replay_buffer = ReplayBuffer(buffer_size)
state_dim = 4

action_dim = env.n_cs
action_bound = 0.5  # 动作最大值
#
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)

return_list, MR = train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, device)
print(MR)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('D3PGSA on DY')
plt.show()
