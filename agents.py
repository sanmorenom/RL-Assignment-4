import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

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
                 epsilon=0.1, entropy_coefficient = 0.01, epochs = 5, rollout_timesteps = 1500, lamb = 0.95):
        super().__init__(env, n_actor_layers, n_critic_layers, gamma, actor_lr, critic_lr)
        # weather to use advantage normalization
        self.adv_norm = adv_norm
        self.entropies = []
        self.entropy_coefficient = entropy_coefficient
        self.states = []
        self.actions = []
        self.rollout_timesteps = rollout_timesteps
        self.lamb = lamb
        self.epsilon = epsilon
        self.epochs = epochs
        # overwrite critic from parent class to implement q-value network
        self.critic = VCritic(self.n_observations,n_critic_layers)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
    
    def __update_actor__(self,old_log_probs ,log_probs ,advantage, returns, entropy):
        # calculate advantages, returns are MC q-val estimates and values are from value network
        #advantages = returns - torch.stack(self.values).detach()
        # normalizing advantages for reducing variance further
        # as seen in the following example: https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
        
        ratios = torch.exp(log_probs - old_log_probs)
        surrogate_loss_1 = ratios * advantage
        surrogate_loss_2 = torch.clamp(ratios, min=1.0-self.epsilon, max=1.0+self.epsilon) * advantage

        # PPO clipped loss
        surr_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
        entropy_bonus = entropy.mean()*self.entropy_coefficient
        loss = torch.sum(-(surr_loss + entropy_bonus))
        # do gradient decent step
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        
    def __select_action__(self, state):
        state = torch.from_numpy(state).float()
        # get action probabilitys and  estimates
        probs = self.actor(state)
        action_dist = Categorical(probs)
        # sample an action using probability estimates
        action = action_dist.sample()
        # also return log probability since we need it for actor updates
        return action.item(), action_dist.log_prob(action)
    
    def __evaluate_actor__(self, states, actions):
        probs = self.actor(torch.tensor(states, dtype=torch.float32))
        action_dist = Categorical(probs)
        return action_dist.log_prob(torch.tensor(actions, dtype=torch.float32)), action_dist.entropy()
    
    def optimize(self, budget):
        """
        Sample episodes and update model.
        Returns evaluation results for every 250 steps in fromat (avg_return, env_steps_taken).
        Budget 
        """
        iterations = budget
        t0 = time()
        evaluation = []
        # sample episodes within a given budget
        while budget>0:
            self.__reset_buffers__()
            state, _  = self.env.reset()

            # sample action from actor (calculate the log prob as well to prevent overhead)
            action, log_prob = self.__select_action__(state)
            terminated = False
            # sample full episode 
            roll_log_probs, roll_acts, roll_states, roll_rews, roll_lens, roll_vals, roll_dones, steps, roll_evaluation = self.__rollout__(budget, iterations)
            budget -= steps
            evaluation.extend(roll_evaluation)
            advantage = self.__calculate_gae__(roll_rews,roll_vals,roll_dones)
            if self.adv_norm:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            values = self.critic(torch.tensor(roll_states, dtype=torch.float32)).squeeze()
            rtgs = advantage + values.detach()   
            

            for _ in range(self.epochs):
                # update both actor and critic
                curr_log_probs,entropy = self.__evaluate_actor__(roll_states, roll_acts)
                self.__update_actor__(roll_log_probs ,curr_log_probs ,advantage, rtgs, entropy)
                self.__update_critic__(rtgs, roll_states)

            # empty the buffers
            self.__reset_buffers__()

            # print current training progress, eta, and current performance (avg evaluation returns)
            progress = (((iterations-budget)/iterations)*100)
            if progress > 0:
                eta = (time()-t0)*((100-progress)/progress)
                #print(f"\rProgress: {progress:.2f}% ETA: {(eta):.0f}s", end='', flush=True)
        print() 
        return evaluation
    def __safe_to_buffer__(self, state, action, reward, log_prob):
        # parant class implements q_net; thus we overwrite it with V_net
        self.values.append(self.critic(torch.tensor(state)).squeeze(-1))
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.states.append(state)
        self.actions.append(action)
    
    
    def __update_critic__(self,returns, states):
        # loss between target value (calculated from returns) and predicted q_vals
        values = self.critic(torch.tensor(states, dtype=torch.float32)).squeeze()
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
    
    def __calculate_gae__(self, rewards, values, dones):
        roll_advantages = []  # List to store computed advantages for each timestep
        # Iterate over each episode's rewards, values, and done flags
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []  # List to store advantages for the current episode
            last_advantage = 0  # Initialize the last computed advantage

            # Calculate episode advantage in reverse order (from last timestep to first)
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    # Calculate the temporal difference (TD) error for the current timestep
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    # Special case at the boundary (last timestep)
                    delta = ep_rews[t] - ep_vals[t]

                # Calculate Generalized Advantage Estimation (GAE) for the current timestep
                advantage = delta + self.gamma * self.lamb * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage  # Update the last advantage for the next timestep
                advantages.insert(0, advantage)  # Insert advantage at the beginning of the list

            # Extend the batch_advantages list with advantages computed for the current episode
            roll_advantages.extend(advantages)

        # Convert the batch_advantages list to a PyTorch tensor of type float
        return torch.tensor(roll_advantages, dtype=torch.float32)

  
  
  
    def __reset_buffers__(self):
        del self.values[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.states[:]
        del self.actions[:]

    def __rollout__(self,budget,iterations):
        # Batch data. For more details, check function header.
        roll_log_probs = []
        roll_acts = []
        roll_states = []
        roll_rews = []
        roll_lens = []
        roll_vals = []
        roll_dones = []
        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        roll_evaluation=[]
        count = 0 # Keeps track of how many timesteps we've run so far this batch
        performance = 0
        # sample episodes within a given budget
        while count<self.rollout_timesteps :
            ep_rews = []
            ep_vals = []
            ep_dones = []
            self.__reset_buffers__()
            state, _  = self.env.reset()

            # sample action from actor (calculate the log prob as well to prevent overhead)
            action, log_prob = self.__select_action__(state)
            terminated = False
            truncated = False
            e_length = 0
            # sample full episode 
            while True:
                ep_dones.append((terminated or truncated))
                roll_states.append(state)
                val = self.critic(torch.tensor(state, dtype=torch.float32))
                roll_log_probs.append(log_prob)
                roll_acts.append(action)
                # take action in env
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                # save info to buffers
                self.__safe_to_buffer__(state,action,reward,log_prob)
                
                next_action, log_prob = self.__select_action__(state)
                ep_rews.append(reward)
                ep_vals.append(val.flatten())
                
                # advance state and action
                state = next_state
                action = next_action
                count +=1
                e_length +=1
                if (count+1) % 250 == 0:
                    performance = self.__evaluate_policy__()
                    roll_evaluation.append((performance,iterations-budget-count))
                    print(f"\rCurrent performance: {(performance):.1f}", end='', flush=True)

                if terminated or truncated:
                    break
            roll_lens.append(e_length)
            roll_rews.append(ep_rews)
            roll_vals.append(ep_vals)
            roll_dones.append(ep_dones)
        roll_states = torch.tensor(roll_states, dtype=torch.float32)
        roll_acts = torch.tensor(roll_acts, dtype=torch.float32)
        roll_log_probs = torch.tensor(roll_log_probs, dtype=torch.float32).flatten()
        return roll_log_probs, roll_acts, roll_states, roll_rews, roll_lens, roll_vals, roll_dones, (count), roll_evaluation
if __name__ == "__main__":

    # quick test run; The episode returns are maximised at roughly 98 since we are using a discount factor 
    print('test')
    env = gym.make("CartPole-v1")
    ppo = PPO(env,2,2,0.99, 0.001,0.001, True)
    ppo.optimize(100000)
