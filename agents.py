import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
from torch.distributions import Categorical
from time import time



class Actor(nn.Module):
    def __init__(self, n_observations, n_actions, n_layers):
        super(Actor, self).__init__()
        self.n_layers = n_layers
        self.a_in = nn.Linear(n_observations, 128)
        for idx in range(self.n_layers):
            setattr(self, f'layer{idx+1}', nn.Linear(128,128))
        self.a_out = nn.Linear(128, n_actions)

    def forward(self, observations):
        x = F.relu(self.a_in(observations))
        for idx in range(self.n_layers):
            x = F.relu(getattr(self, 'layer%d' % (idx+1))(x))
        return F.softmax(self.a_out(x), dim=-1)
    
class QCritic(nn.Module):
    def __init__(self, n_observations, n_actions, n_layers):
        super(QCritic, self).__init__()
        self.n_layers = n_layers
        self.c_in = nn.Linear(n_observations, 128)
        for idx in range(self.n_layers):
            setattr(self, f'layer{idx+1}', nn.Linear(128,128))
        self.c_out = nn.Linear(128, n_actions)

    def forward(self, observations):
        x = F.relu(self.c_in(observations))
        for idx in range(self.n_layers):
            x = F.relu(getattr(self, 'layer%d' % (idx+1))(x))
        return self.c_out(x)
    
class VCritic(nn.Module):
    def __init__(self, n_observations, n_layers):
        super(VCritic, self).__init__()
        self.n_layers = n_layers
        self.c_in = nn.Linear(n_observations, 128)
        for idx in range(self.n_layers):
            setattr(self, f'layer{idx+1}', nn.Linear(128,128))
        self.c_out = nn.Linear(128, 1)

    def forward(self, observations):
        x = F.relu(self.c_in(observations))
        for idx in range(self.n_layers):
            x = F.relu(getattr(self, 'layer%d' % (idx+1))(x))
        return self.c_out(x)
    
class ModelFreeLearner():
    """
    ModelFreeLearner serves as a parent class for the REINFORCE, ActorCritic (AC) and
    Advantage Actor Critic (A2C). 
    This design choice prevents code duplication since the core functionalities in the 
    optimization/training step is the same for each learner.  
    """
    def __init__(self, env:gym.Env, n_actor_layers, n_critic_layers, gamma, actor_lr, critic_lr):
        # initialize environment and exrtract state and action space
        self.env = env
        self.n_actions = self.env.action_space.n
        state,info = self.env.reset()
        self.n_observations = len(state)

        # initialize hyperparameters
        self.gamma = gamma

        # initialize actor and critic functions
        self.actor = Actor(self.n_observations,self.n_actions,n_actor_layers)
        self.critic = QCritic(self.n_observations,self.n_actions,n_critic_layers)

        # initialize buffers to safe rewards during episodes
        self.values = []
        self.log_probs = []
        self.rewards = []

        # initilize optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def __select_action__(self, state):
        state = torch.from_numpy(state).float()
        # get action probabilitys and  estimates
        probs = self.actor(state)
        action_dist = Categorical(probs)
        # sample an action using probability estimates
        action = action_dist.sample()
        # also return log probability since we need it for actor updates
        return action.item(), action_dist.log_prob(action)
    
    def __select_action_mode__(self, state):
        state = torch.from_numpy(state).float()
        # get action probabilitys and  estimates
        probs = self.actor(state)
        action_dist = Categorical(probs)
        # Get the mode of the distribution to simulate greedy action selection
        action = action_dist.mode
        return action.item()
    
    def __reset_buffers__(self):
        del self.values[:]
        del self.log_probs[:]
        del self.rewards[:]

    def __update_actor__(self, returns):
        pass

    def __update_critic__(self,returns):
        # loss between target value (calculated from returns) and predicted q_vals
        loss = F.mse_loss(torch.stack(self.values), returns.detach())
        # do gradient decent step
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def __get_returns__(self):
        returns = []
        R = 0 
        # iterate over reversed rewards array 
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0,R)
        return torch.as_tensor(returns, dtype=torch.float32)
    
    def __safe_to_buffer__(self,state,action,reward,log_prob):
        self.values.append(self.critic(torch.tensor(state))[action])
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def optimize(self, budget):
        """
        Sample episodes and update model.
        Returns evaluation results for every 250 steps in fromat (avg_return, env_steps_taken).
        Budget 
        """
        iterations = budget
        t0 = time()
        performance = 0
        evaluation = []
        # sample episodes within a given budget
        while budget>0:
            self.__reset_buffers__()
            state, _  = self.env.reset()

            # sample action from actor (calculate the log prob as well to prevent overhead)
            action, log_prob = self.__select_action__(state)
        
            terminated = False
            # sample full episode 
            while True:
                # take action in env
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                # save info to buffers
                self.__safe_to_buffer__(state,action,reward,log_prob)

                next_action, log_prob = self.__select_action__(state)
                # advance state and action
                state = next_state
                action = next_action
 
                budget -=1

                if budget % 250 == 0:
                    performance = self.__evaluate_policy__()
                    evaluation.append((performance,iterations-budget))

                if terminated or truncated:
                    break

            # calculate returns based on rewards 
            returns = self.__get_returns__()

            # update both actor and critic
            self.__update_actor__(returns)
            self.__update_critic__(returns)

            # empty the buffers
            self.__reset_buffers__()

            # print current training progress, eta, and current performance (avg evaluation returns)
            progress = (((iterations-budget)/iterations)*100)
            eta = (time()-t0)*((100-progress)/progress)
            print(f"\rProgress: {progress:.2f}% ETA: {(eta):.0f}s Current performance: {(performance):.1f}", end='', flush=True)
        print() 
        return evaluation
    
    def __evaluate_policy__(self):
        "Evaluates the current policy over 3 episodes and returns average return"
        episode_return = 0
        eval_env = gym.make(self.env.spec)
        for i in range(3):
            state, _  = eval_env.reset()
            terminated = False
            while True:
                #using the mode of the action distribution for greedy action selection
                action = self.__select_action_mode__(state)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                state = next_state
                episode_return += reward 
                if terminated or truncated:
                    break
        return episode_return/3

### Agent implementation ###

class REINFORCE(ModelFreeLearner):
    def __init__(self, env, n_actor_layers, n_critic_layers, gamma, actor_lr, critic_lr):
        super().__init__(env, n_actor_layers, n_critic_layers, gamma, actor_lr, critic_lr)
    def __update_actor__(self, returns):
        # Calculate loss; negative for gradient ascent
        loss = torch.sum(-torch.stack(self.log_probs) * returns.detach())
        # do gradient decent step
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    # override functions to prevent computational overhead since critic is not needed 
    def __update_critic__(self, returns):
        pass
    def __get_deltas__(self, state, action, reward, next_state, next_action):
        pass

class AC(ModelFreeLearner):
    def __init__(self, env, n_actor_layers, n_critic_layers, gamma, actor_lr, critic_lr):
        super().__init__(env, n_actor_layers, n_critic_layers, gamma, actor_lr, critic_lr)
    
    def __update_actor__(self, returns):
        # Calculate loss; negative for gradient ascent 
        loss = torch.sum(-torch.stack(self.log_probs) * torch.stack(self.values).detach())
        # do gradient decent step
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()


class A2C(ModelFreeLearner):
    def __init__(self, env, n_actor_layers, n_critic_layers, gamma, actor_lr, critic_lr, adv_norm = False):
        super().__init__(env, n_actor_layers, n_critic_layers, gamma, actor_lr, critic_lr)
        # weather to use advantage normalization
        self.adv_norm = adv_norm
        # overwrite critic from parent class to implement q-value network
        self.critic = VCritic(self.n_observations,n_critic_layers)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
    
    def __update_actor__(self, returns):
        # calculate advantages, returns are MC q-val estimates and values are from value network
        advantages = returns - torch.stack(self.values).detach()
        # normalizing advantages for reducing variance further
        # as seen in the following example: https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
        if self.adv_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) 
        loss = torch.sum(-torch.stack(self.log_probs) * advantages)
        # do gradient decent step
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
    
    def __safe_to_buffer__(self, state, action, reward, log_prob):
        # parant class implements q_net; thus we overwrite it with V_net
        self.values.append(self.critic(torch.tensor(state)).squeeze(-1))
        self.rewards.append(reward)
        self.log_probs.append(log_prob)


class PPO(ModelFreeLearner):
    def __init__(self, env, n_actor_layers, n_critic_layers, gamma, actor_lr, critic_lr, adv_norm = False, 
                 epsilon=0.2, entropy_coefficient = 0.01, epochs = 4,minibatch_len=100, rollout_len=2000, target_kl=0.02):
        super().__init__(env, n_actor_layers, n_critic_layers, gamma, actor_lr, critic_lr)
        # weather to use advantage normalization
        self.adv_norm = adv_norm
        self.entropy_coefficient = entropy_coefficient
        self.states = []
        self.actions = []
        self.dones = []
        self.minibatch_len = minibatch_len
        self.rollout_len = rollout_len
        self.epsilon = epsilon
        self.target_kl = target_kl
        self.epochs = epochs
        # overwrite critic from parent class to implement q-value network
        self.critic = VCritic(self.n_observations,n_critic_layers)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
    def __get_returns_roll__(self,last_value):
        returns = []
        R = last_value
        # iterate over reversed rewards array 
        for r in reversed(range(len(self.rewards))):
            if self.dones[r]:
                R = 0.0
            R = self.rewards[r] + self.gamma * R
            returns.insert(0,R)
        return torch.as_tensor(returns, dtype=torch.float32)
    def __update_actor__(self,old_log_probs ,log_probs ,advantage, returns, entropy):
        
        # normalizing advantages for reducing variance further
        # as seen in the following example: https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
        log_ratios = log_probs - old_log_probs
        ratios = torch.exp(log_ratios)
        approx_kl = ((ratios - 1.0) - log_ratios).mean().detach().item()
        surrogate_loss_1 = ratios * advantage
        surrogate_loss_2 = torch.clamp(ratios, min=1.0-self.epsilon, max=1.0+self.epsilon) * advantage

        # PPO clipped loss
        surr_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
        entropy_bonus = entropy.mean()*self.entropy_coefficient
        loss = torch.sum(-(surr_loss.mean() + entropy_bonus))
        # do gradient decent step
        self.actor_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()
        return approx_kl
    
    def __select_action__(self, state):
        state = torch.from_numpy(state).float()
        # get action probabilitys and  estimates
        probs = self.actor(state)
        action_dist = Categorical(probs)
        # sample an action using probability estimates
        action = action_dist.sample()
        # also return log probability since we need it for actor updates
        return action.item(), action_dist.log_prob(action)
    
    def __evaluate_actor__(self):
        probs = self.actor(torch.tensor(self.states, dtype=torch.float32))
        action_dist = Categorical(probs)
        return action_dist.log_prob(torch.tensor(self.actions, dtype=torch.float32)), action_dist.entropy()
    
    def __evaluate_actor__(self,states,actions):
        probs = self.actor(states)
        action_dist = Categorical(probs)
        return action_dist.log_prob(actions), action_dist.entropy()
    
    def optimize(self, budget):
        """
        Sample episodes and update model.
        Returns evaluation results for every 250 steps in fromat (avg_return, env_steps_taken).
        Budget 
        """
        iterations = budget
        t0 = time()
        performance = 0
        evaluation = []
        # sample episodes within a given budget
        while budget>0:
            step_counter = 0
            self.__reset_buffers__()
            while step_counter<self.rollout_len:
                state, _  = self.env.reset()

                # sample action from actor (calculate the log prob as well to prevent overhead)
                action, log_prob = self.__select_action__(state)
                terminated = False
                # sample full episode 
                while True:
                    # take action in env
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    # save info to buffers
                    self.__safe_to_buffer__(state,action,reward,log_prob,done)
                    state = next_state
                    next_action, log_prob = self.__select_action__(state)
                    # advance state and action
                    
                    action = next_action
    
                    budget -=1
                    step_counter += 1
                    if budget % 250 == 0:
                        performance = self.__evaluate_policy__()
                        evaluation.append((performance,iterations-budget))

                    if done:
                        break

            # calculate returns based on rewards 
            with torch.no_grad():
                if len(self.dones) > 0 and not self.dones[-1]:
                    final_state= torch.tensor(state, dtype=torch.float32)
                    last_value_tensor = self.critic(final_state).squeeze(-1)
                    last_value = last_value_tensor.item()
                else:
                    last_value = 0.0
            returns = self.__get_returns_roll__(last_value)
            advantage = self.__get_advantage__(returns)
            roll_states = torch.tensor(np.array(self.states), dtype=torch.float32)
            roll_actions = torch.tensor(self.actions, dtype=torch.long)
            old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).detach()
            batch_size = roll_states.shape[0]
            last_approx_kl = 0.0
            stop_update = False
            for _ in range(self.epochs):
                idx = torch.randperm(batch_size)
                for i in range(0, batch_size, self.minibatch_len):
                    end = i + self.minibatch_len
                    batch_idx = idx[i:end]
                    batch_states = roll_states[batch_idx]
                    batch_actions = roll_actions[batch_idx]
                    batch_old_log_probs = old_log_probs[batch_idx]
                    batch_returns = returns[batch_idx]
                    batch_advantage = advantage[batch_idx]
                    # update both actor and critic
                    curr_log_probs,entropy = self.__evaluate_actor__(batch_states,batch_actions)
                    approx_kl = self.__update_actor__(batch_old_log_probs ,curr_log_probs ,batch_advantage, batch_returns, entropy)
                    self.__update_critic__(batch_returns,batch_states)
                    last_approx_kl = approx_kl
                    if self.target_kl is not None and approx_kl > self.target_kl:
                        stop_update = True
                        break
                if stop_update:
                    break

            # empty the buffers
            self.__reset_buffers__()

            # print current training progress, eta, and current performance (avg evaluation returns)
            progress = (((iterations-budget)/iterations)*100)
            eta = (time()-t0)*((100-progress)/progress)
            print(f"\rProgress: {progress:.2f}% ETA: {(eta):.0f}s Current performance: {(performance):.1f}", end='', flush=True)
        print() 
        return evaluation
    def __safe_to_buffer__(self, state, action, reward, log_prob,done):
        # parant class implements q_net; thus we overwrite it with V_net
        self.values.append(self.critic(torch.tensor(state)).squeeze(-1))
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.states.append(state)
        self.actions.append(action)
        self.dones.append(done)
    
    def __update_critic__(self,returns):
        # loss between target value (calculated from returns) and predicted q_vals
        values = self.critic(torch.tensor(self.states, dtype=torch.float32)).squeeze()
        loss = F.mse_loss(values, returns.detach())
        # do gradient decent step
        self.critic_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()
    
    def __update_critic__(self,returns,states):
        # loss between target value (calculated from returns) and predicted q_vals
        values = self.critic(states).squeeze()
        loss = F.mse_loss(values, returns.detach())
        # do gradient decent step
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        
    def __get_advantage__(self, returns):
        values = self.critic(torch.tensor(self.states, dtype=torch.float32)).squeeze()
        advantage = returns - values.detach()
        # normalizing advantages for reducing variance further
        # as seen in the following example: https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
        if self.adv_norm:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8) 
        return advantage
    
    def __reset_buffers__(self):
        del self.values[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.states[:]
        del self.actions[:]
        del self.dones[:]

if __name__ == "__main__":

    # quick test run; The episode returns are maximised at roughly 98 since we are using a discount factor 
    print('test')
    env = gym.make("CartPole-v1")
    ppo = PPO(env,2,2,0.99, 0.001,0.001, True)
    ppo.optimize(100000)
