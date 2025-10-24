# Minimal A3C implementation for CartPole-v1 (PyTorch)
# Usage: python a3c.py --workers 4
# Requirements: torch, gym
# 2025/10/24 

import argparse
import time
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

# ----------------------------
# Actor-Critic network
# ----------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, action_space):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # actor head: outputs logits for actions
        self.pi = nn.Linear(hidden_size, action_space)
        # critic head: outputs state value
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.pi(x)
        value = self.v(x)
        return logits, value

# ----------------------------
# Shared Adam optimizer
# ----------------------------
class SharedAdam(torch.optim.Adam):
    """Adam optimizer with shared states for multiprocessing."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # move optimizer state tensors to shared memory
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

# ----------------------------
# Gym reset compatibility helper
# ----------------------------
def env_reset(env):
    out = env.reset()
    # gym v0.26+ returns (obs, info)
    if isinstance(out, tuple):
        return out[0]
    return out

# ----------------------------
# Worker process
# ----------------------------
class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_ep, global_ep_r, result_queue, opts):
        super(Worker, self).__init__()
        self.global_net = global_net
        self.optimizer = optimizer
        self.global_ep = global_ep
        self.global_ep_r = global_ep_r
        self.result_queue = result_queue
        self.opts = opts
        self.name = f"worker-{self.opts.worker_id}"
        # local network (not shared)
        self.local_net = ActorCritic(self.opts.state_size, self.opts.hidden_size, self.opts.action_space)

    def run(self):
        env = gym.make(self.opts.env_name)
        torch.manual_seed(self.opts.worker_id)
        while True:
            # termination condition
            with self.global_ep.get_lock():
                if self.global_ep.value >= self.opts.max_episodes:
                    break

            state = env_reset(env)
            if isinstance(state, tuple):
                state = state[0]
            state = torch.from_numpy(state).float()
            buffer_states = []
            buffer_actions = []
            buffer_rewards = []
            episode_reward = 0.0
            done = False

            while not done:
                # sync local network with global network
                self.local_net.load_state_dict(self.global_net.state_dict())

                # run up to t_max steps or until done
                for _ in range(self.opts.t_max):
                    logits, value = self.local_net(state.unsqueeze(0))
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample().item()

                    step_out = env.step(action)
                    # compatibility for gym step output shapes
                    if len(step_out) == 5:
                        next_state, reward, terminated, truncated, _ = step_out
                        done_flag = terminated or truncated
                    else:
                        next_state, reward, done_flag, _ = step_out

                    # handle env returning (obs, info) in next_state
                    if isinstance(next_state, tuple):
                        next_state = next_state[0]

                    next_state_t = torch.from_numpy(next_state).float()
                    episode_reward += reward

                    buffer_states.append(state)
                    buffer_actions.append(action)
                    buffer_rewards.append(reward)

                    state = next_state_t

                    if done_flag:
                        done = True
                        break

                # compute n-step returns
                if done:
                    R = 0.0
                else:
                    with torch.no_grad():
                        _, v_next = self.local_net(state.unsqueeze(0))
                        R = v_next.item()

                returns = []
                for r in reversed(buffer_rewards):
                    R = r + self.opts.gamma * R
                    returns.insert(0, R)
                returns = torch.tensor(returns, dtype=torch.float32)

                # prepare tensors
                if len(buffer_states) == 0:
                    break
                states_tensor = torch.stack(buffer_states)
                actions_tensor = torch.tensor(buffer_actions, dtype=torch.int64)

                # compute losses on local net
                logits, values = self.local_net(states_tensor)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(actions_tensor)
                entropy = dist.entropy().mean()
                values = values.squeeze(1)

                advantages = returns - values.detach()
                policy_loss = -(log_probs * advantages).mean()
                value_loss = F.mse_loss(values, returns)
                total_loss = policy_loss + self.opts.value_loss_coef * value_loss - self.opts.entropy_coef * entropy

                # backprop on local net
                self.optimizer.zero_grad()
                total_loss.backward()

                # push local gradients to the global network
                for local_param, global_param in zip(self.local_net.parameters(), self.global_net.parameters()):
                    if global_param.grad is None:
                        # clone gradient tensor to avoid sharing the local tensor
                        global_param.grad = local_param.grad.clone()
                    else:
                        global_param.grad.copy_(local_param.grad)

                # step global optimizer
                self.optimizer.step()

                # clear buffers
                buffer_states.clear()
                buffer_actions.clear()
                buffer_rewards.clear()

                if done:
                    # update global episode counters and moving average reward
                    with self.global_ep.get_lock():
                        self.global_ep.value += 1
                    with self.global_ep_r.get_lock():
                        if self.global_ep_r.value == 0.0:
                            self.global_ep_r.value = episode_reward
                        else:
                            # exponential moving average
                            self.global_ep_r.value = self.global_ep_r.value * 0.99 + episode_reward * 0.01

                    print(f"{self.name} | Episode: {self.global_ep.value} | EpisodeReward: {episode_reward:.2f} | AvgReward: {self.global_ep_r.value:.2f}")
                    self.result_queue.put(episode_reward)
                    break

        env.close()
        self.result_queue.put(None)

# ----------------------------
# Main training function
# ----------------------------
def train(opts):
    mp.set_start_method('spawn', force=True)

    # create a temporary env to get sizes
    env = gym.make(opts.env_name)
    obs = env_reset(env)
    if isinstance(obs, tuple):
        obs = obs[0]
    state_size = obs.shape[0]
    action_space = env.action_space.n
    env.close()

    opts.state_size = state_size
    opts.action_space = action_space

    global_net = ActorCritic(state_size, opts.hidden_size, action_space)
    global_net.share_memory()
    optimizer = SharedAdam(global_net.parameters(), lr=opts.lr)

    global_ep = mp.Value('i', 0)
    global_ep_r = mp.Value('d', 0.0)
    result_queue = mp.Queue()

    workers = []
    for i in range(opts.workers):
        wopts = argparse.Namespace(**vars(opts))
        wopts.worker_id = i
        w = Worker(global_net, optimizer, global_ep, global_ep_r, result_queue, wopts)
        w.start()
        workers.append(w)

    # collect results
    rewards = []
    finished_workers = 0
    while True:
        r = result_queue.get()
        if r is None:
            finished_workers += 1
            if finished_workers == opts.workers:
                break
        else:
            rewards.append(r)

    for w in workers:
        w.join()

    print("Training finished. Episodes collected:", len(rewards))

# ----------------------------
# CLI
# ----------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default='CartPole-v1', help='Gym environment name')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel worker processes')
    parser.add_argument('--max-episodes', type=int, default=500, help='Maximum number of training episodes (global)')
    parser.add_argument('--t-max', type=int, default=5, help='Number of steps before each update (n-step)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for SharedAdam')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='Value loss coefficient')
    return parser.parse_args()

if __name__ == '__main__':
    opts = get_args()
    start = time.time()
    try:
        train(opts)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        print("Elapsed time: {:.2f} seconds".format(time.time() - start))
