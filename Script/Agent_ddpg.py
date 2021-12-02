# submission을 위한 Agent 파일 입니다.
# policy(), save_model(), load_model()의 arguments와 return은 변경하실 수 없습니다.
# 그 외에 자유롭게 코드를 작성 해주시기 바랍니다.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from Drone_gym import Drone_gym
import random

from keras.models import *
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Lambda
from keras.optimizers import Adam
from keras.initializers import RandomUniform
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


## 이미지 관련 코드 
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray

class Q_network(nn.Module):
    def __init__(self):
        super(Q_network, self).__init__()

        self.vector_fc = nn.Linear(95, 512) # embedding vector observation
        
        self.image_cnn = nn.Sequential(
            nn.Conv2d(4,32,kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.ReLU()
            )
        self.image_fc = nn.Linear(1024,512) ## embedding image observation
    
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)

        self.fc_action1 = nn.Linear(512, 256)
        self.fc_action2 = nn.Linear(256, 27)

    def forward(self,x):
        
        front_obs, right_obs, rear_obs, left_obs, btn_obs, vector_obs = x
        batch = front_obs.size(0)

        vector_obs = self.vector_fc(vector_obs).view(batch,-1) 
        
        front_obs = self.image_fc(self.image_cnn(front_obs).view(batch,-1)) 
        right_obs = self.image_fc(self.image_cnn(right_obs).view(batch,-1))
        rear_obs = self.image_fc(self.image_cnn(rear_obs).view(batch,-1))
        left_obs = self.image_fc(self.image_cnn(left_obs).view(batch,-1))
        btn_obs = self.image_fc(self.image_cnn(btn_obs).view(batch,-1))

        image_obs = (front_obs + right_obs + rear_obs + left_obs + btn_obs)
        
        x = torch.cat((vector_obs,image_obs),-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        action_logit = torch.relu(self.fc_action1(x))
        Q_values  = self.fc_action2(action_logit)

        return Q_values 
     
class Agent:
    def __init__(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.policy_net = Q_network().to(device)
        self.Q_target_net = Q_network().to(device)
        self.learning_rate = 0.00025

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.Q_target_net.load_state_dict(self.policy_net.state_dict()) ## Q의 뉴럴넷 파라미터를 target에 복사
        
        self.epsilon = 1 ## epsilon 1 부터 줄어들어감.
        self.epsilon_decay = 0.00001 ## episilon 감쇠 값.
        
        self.device = device
        self.data_buffer = deque(maxlen=100000)

        self.init_check = 0
        self.cur_state = None
        
    def epsilon_greedy(self, Q_values):
        if np.random.rand() <= self.epsilon: ## 0~1의 균일분포 표준정규분포 난수를 생성. 정해준 epsilon 값보다 작은 random 값이 나오면 
            #action = random.randrange(27) ## action을 random하게 선택합니다.
            action = random.randrange(7)
            return action
        
        else: ## epsilon 값보다 크면, 학습된 Q_player NN 에서 얻어진 Q_values 값중 가장 큰 action을 선택합니다.
            return Q_values.argmax().item()
        
    def policy(self, obs): 
        """Policy function p(a|s), Select three actions.

        Args:
            obs: all observations. consist of 6 observations.
                   (front image observation, right image observation, 
                   rear image observation, left image observation, 
                   bottom observation, vector observation)

        Return:
            3 continous actions vector (range -1 ~ 1) from policy.
            ex) [-1~1, -1~1, -1~1]
        """
        if self.init_check == 0:
            self.cur_obs = self.init_obs(obs)
            self.init_check += 1
        else:
            self.cur_obs = self.accumulated_all_obs(self.cur_obs, obs)
        Q_values = self.policy_net(self.torch_obs( self.cur_obs))
        action = self.epsilon_greedy(Q_values)
     
        return self.convert_action(action) ## inference를 위한 (x, y, z) 위치 action, 각각 0~1 사이의 continuous value.

    def train_policy(self, obs): 
        """Policy function p(a|s), Select three actions.

        Args:
            obs: all observations. consist of 6 observations.
                   (front image observation, right image observation, 
                   rear image observation, left image observation, 
                   bottom observation, vector observation)

        Return:
            3 continous actions vector (range -1 ~ 1) from policy.
            ex) [-1~1, -1~1, -1~1]
        """
        Q_values = self.policy_net(obs)
        action = self.epsilon_greedy(Q_values)
     
        return self.convert_action(action), action
    
    def save_model(self):
        torch.save(self.policy_net.state_dict(), './best_model/best_model.pt')
        return None

    def load_model(self):
        self.policy_net.load_state_dict(torch.load('./best_model/best_model.pt', map_location=self.device))
        return None

    def store_trajectory(self, traj):
        """
           store data
        """
        self.data_buffer.append(traj)

    def re_scale_frame(self, obs):
        """
        change rgb to gray.
        """
        return resize(rgb2gray(obs),(64,64))

    def init_image_obs(self, obs):
        """
        set initial observation s_0, stacked 4 s_0 frames.
        """
        obs = self.re_scale_frame(obs)
        frame_obs = [obs for _ in range(4)]
        frame_obs = np.stack(frame_obs, axis=0)

        return frame_obs

    def init_obs(self, obs):
        """
           set initial observation
        """
        front_obs = self.init_image_obs(obs[0])
        right_obs = self.init_image_obs(obs[1])
        rear_obs = self.init_image_obs(obs[2])
        left_obs = self.init_image_obs(obs[3])
        btn_obs = self.init_image_obs(obs[4])
        vector_obs = obs[5]
        
        return (front_obs,right_obs,rear_obs,left_obs,btn_obs,vector_obs)

    def torch_obs(self, obs):
        """
            convert to torch tensor
        """
        front_obs = torch.Tensor(obs[0]).unsqueeze(0).to(self.device)
        right_obs = torch.Tensor(obs[1]).unsqueeze(0).to(self.device)
        rear_obs = torch.Tensor(obs[2]).unsqueeze(0).to(self.device)
        left_obs = torch.Tensor(obs[3]).unsqueeze(0).to(self.device)
        btn_obs = torch.Tensor(obs[4]).unsqueeze(0).to(self.device)
        vector_obs = torch.Tensor(obs[5]).to(self.device) # 16
  
        return (front_obs,right_obs,rear_obs,left_obs,btn_obs,vector_obs)

    def accumulated_image_obs(self, obs, new_frame):
        """
            accumulated image observation.
        """
        temp_obs = obs[1:,:,:]
        new_frame = np.expand_dims(self.re_scale_frame(new_frame), axis=0)
        frame_obs = np.concatenate((temp_obs, new_frame),axis=0)
    
        return frame_obs

    def accumulated_all_obs(self, obs, next_obs): 
        """
            accumulated all observation.
        """
        front_obs = self.accumulated_image_obs(obs[0], next_obs[0])
        right_obs = self.accumulated_image_obs(obs[1], next_obs[1])        
        rear_obs = self.accumulated_image_obs(obs[2], next_obs[2])
        left_obs = self.accumulated_image_obs(obs[3], next_obs[3])
        btn_obs = self.accumulated_image_obs(obs[4], next_obs[4])
        vector_obs = next_obs[5]
     
        return (front_obs,right_obs,rear_obs,left_obs,btn_obs,vector_obs)
    
    #상하좌우앞뒤, 하강 -2까지
    def convert_action2(self, action):
        if action == 0:
            return [1, 0, 0]
        elif action == 1:
            return [-1, 0, 0]
        elif action == 2:
            return [0, 1, 0]
        elif action == 3:
            return [0, -1, 0]
        elif action == 4:
            return [0, 0, 1]
        elif action == 5:
            return [0, 0, -2]
        elif action == 6:
            return [0, 0, -1]
                
    def convert_action(self, action):
        if action == 0:
            return [-1, -1, -1]
        elif action == 1:
            return [-1, -1,  0]
        elif action == 2:
            return [-1, -1,  1]
        elif action == 3:
            return [-1,  0, -1]
        elif action == 4:
            return [-1,  0,  0]
        elif action == 5:
            return [-1,  0,  1]
        elif action == 6:
            return [-1,  1, -1]
        elif action == 7:
            return [-1,  1,  0]
        elif action == 8:
            return [-1,  1,  1]
        elif action == 9:
            return [ 0, -1, -1]
        elif action == 10:
            return [ 0, -1,  0]
        elif action == 11:
            return [ 0, -1,  1]
        elif action == 12:
            return [ 0,  0, -1]
        elif action == 13:
            return [ 0,  0,  0]
        elif action == 14:
            return [ 0,  0,  1]
        elif action == 15:
            return [ 0,  1, -1]
        elif action == 16:
            return [ 0,  1,  0]
        elif action == 17:
            return [ 0,  1,  1]
        elif action == 18:
            return [ 1, -1, -1]
        elif action == 19:
            return [ 1, -1,  0]
        elif action == 20:
            return [ 1, -1,  1]
        elif action == 21:
            return [ 1,  0, -1]
        elif action == 22:
            return [ 1,  0,  0]
        elif action == 23:
            return [ 1,  0,  1]
        elif action == 24:
            return [ 1,  1, -1]
        elif action == 25:
            return [ 1,  1,  0]
        elif action == 26:
            return [ 1,  1,  1]

    def batch_torch_obs(self, obs):
        front_obs = torch.Tensor(np.stack([s[0] for s in obs], axis=0)).to(self.device)
        right_obs = torch.Tensor(np.stack([s[1] for s in obs], axis=0)).to(self.device)      
        rear_obs = torch.Tensor(np.stack([s[2] for s in obs], axis=0)).to(self.device)
        left_obs = torch.Tensor(np.stack([s[3] for s in obs], axis=0)).to(self.device)
        btn_obs = torch.Tensor(np.stack([s[4] for s in obs], axis=0)).to(self.device)
        vector_obs = torch.Tensor(np.stack([s[5] for s in obs], axis=0)).to(self.device)
        
        return (front_obs,right_obs,rear_obs,left_obs,btn_obs,vector_obs)

    def update_target(self):
        self.Q_target_net.load_state_dict(self.policy_net.state_dict()) ## Q_player NN에 학습된 weight를 그대로 Q_target에 복사함.

    def train(self):
        
        gamma = 0.99

        self.epsilon -= self.epsilon_decay 
        self.epsilon = max(self.epsilon, 0.05) 
        random_mini_batch = random.sample(self.data_buffer, 32) ## batch_size 만큼 random sampling.
        
        # data 분배
        obs_list, action_list, reward_list, next_obs_list, mask_list = [], [], [], [], []
        
        for all_obs in random_mini_batch:
            s, a, r, next_s, mask = all_obs
            obs_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_obs_list.append(next_s)
            mask_list.append(mask)
        
        # tensor.
        obses = self.batch_torch_obs(obs_list)
        actions = torch.LongTensor(action_list).unsqueeze(1).to(self.device)
        rewards = torch.Tensor(reward_list).to(self.device)
        next_obses = self.batch_torch_obs(next_obs_list)
        masks = torch.Tensor(mask_list).to(self.device)

        # get Q-value
        Q_values = self.policy_net(obses)
        q_value = Q_values.gather(1, actions).view(-1)
        
        # get target
        target_q_value = self.Q_target_net(next_obses).max(1)[0] 

        Y = rewards + masks * gamma * target_q_value

        # loss 정의 
        MSE = torch.nn.MSELoss() 
        loss = MSE(q_value, Y.detach())

        # backward 시작!
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class RLAgent:

    def __init__(self, env):
        ENV_NAME = 'drone'
        # Get the environment and extract the number of actions.
        #env = gym.make(ENV_NAME)
        self.env = env
        np.random.seed(123)
        self.env.seed(123)
        assert len(self.env.action_space.shape) == 1
        nb_actions = self.env.action_space.shape[0]

        # Next, we build a very simple model.
        self.actor = Sequential()
        self.actor.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        self.actor.add(Dense(16))
        self.actor.add(Activation('relu'))
        self.actor.add(Dense(16))
        self.actor.add(Activation('relu'))
        self.actor.add(Dense(16))
        self.actor.add(Activation('relu'))
        self.actor.add(Dense(nb_actions,activation='tanh',kernel_initializer=RandomUniform()))
        self.actor.add(Lambda(lambda x: x * 60.0))
        print(self.actor.summary())

        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + self.env.observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Concatenate()([action_input, flattened_observation])
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        print(critic.summary())

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
        self.agent = DDPGAgent(nb_actions=nb_actions, actor=self.actor, critic=critic, critic_action_input=action_input,
                          memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                          random_process=random_process, gamma=.99, target_model_update=1e-3)
        self.agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
        
        
        
def main():    
    env = Drone_gym(
            time_scale=1.0,
            port=11001,
            filename='../RL_Drone/DroneDelivery.exe') 
    '''
    해상도 변경을 원할 경우, width, height 값 조절.
    env = Drone_gym(
            time_scale=1.0,
            port=11000,
            width=84, height=84, filename='../RL_Drone/DroneDelivery.exe')
    '''
    agent = RLAgent(env)
    agent.agent.fit(env, nb_steps=100000, visualize=True, verbose=1, nb_max_episode_steps=10)

    #After training is done, we save the final weights.
    agent.agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    
    
if __name__ == '__main__':
    main()
