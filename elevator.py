import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DQN
time_check=[]
class Elevator(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        #observation space
        # states0: floor_0_up
        # states1: floor_1_up
        # states2: floor_1_down
        # states3: floor_2_up
        # states4: floor_2_down
        # states5: floor_3_down
        # states6: occupancy
        # states7: position
        super(Elevator, self).__init__()
        self.done = 0
        self.reward = 0
        self.states = np.zeros(8)
        self.states[0]=1
        self.last_time = 0
        self.time = 0
        self.max_occupancy = 5
        self.action_space = spaces.Discrete(3)  # 0 stop, 1 up, 2 down
        self.observation_space = spaces.MultiDiscrete([2,2,2,2,2,2,6,4])

    def reset(self):
        self.states = np.zeros(8)
        #suppose that there are already 2 people
        # waiting on the first floor at the beginning of the session
        self.states[0]=1
        self.last_time = 0
        self.time = 0
        self.floor_0_waiting = 2
        self.floor_0_waiting_list = [1,2]
        self.floor_1_waiting = 0
        self.floor_1_waiting_list = []
        self.floor_2_waiting = 0
        self.floor_2_waiting_list = []
        self.floor_3_waiting = 0
        self.floor_3_waiting_list = []
        self.inside_list = []
        self.done = 0
        self.reward = 0
        return self.states

    def timecheck(self):
        if (self.last_time < 5) & (self.time >= 5):
            self.floor_1_waiting = 5
            self.floor_1_waiting_list.extend(
                np.random.choice([0, 2, 3], size=self.floor_1_waiting,
                                 p=[0.6, 0.2, 0.2]).tolist())
            self.states[1] = 1
            self.states[2] = 1

        elif (self.last_time < 60) & (self.time >= 60):
            self.floor_2_waiting = 5
            self.floor_2_waiting_list.extend(
                np.random.choice([0, 1, 3], size=self.floor_1_waiting, p=[0.6, 0.2, 0.2]).tolist())
            self.states[3] = 1
            self.states[4] = 1
        elif (self.last_time < 120) & (self.time >= 120):
            self.floor_3_waiting = 5
            self.floor_3_waiting_list.extend(
                np.random.choice([0, 1, 2], size=self.floor_1_waiting, p=[0.6, 0.2, 0.2]).tolist())
            self.states[5] = 1
        if (self.time - self.last_time >= 60) and (self.time < 180):
            self.floor_0_waiting_list.extend(
                np.random.choice([1, 2, 3], size=np.random.choice(3, p=[0.9, 0.05, 0.05])).tolist())
        self.last_time = self.time

    def waiting_list_check(self):
        if len(self.floor_0_waiting_list) == 0:
            self.states[0] = 0
        if 0 not in self.floor_1_waiting_list:
            self.states[2] = 0
        if (2 not in self.floor_1_waiting_list) & (3 not in self.floor_1_waiting_list):
            self.states[1] = 0
        if (0 not in self.floor_2_waiting_list) & (1 not in self.floor_2_waiting_list):
            self.states[4] = 0
        if 3 not in self.floor_2_waiting_list:
            self.states[3] = 0
        if len(self.floor_3_waiting_list) == 0:
            self.states[5] = 0

    def done_check(self):
        if (self.states[0] == 0) & (self.states[2] == 0) & (
                self.states[1] == 0) & (self.states[4] == 0) & (
                self.states[3] == 0) & (self.states[5] == 0) &(self.states[6] == 0):
            self.done=1
            time_check.append(self.time)
        elif self.time>=900:
            self.done=1
            print("More than 15 minutes")
        return self.done

    def rotating_people(self):
        self.inside_list = [x for x in self.inside_list if x != self.states[7]]
        remaning_places = self.max_occupancy - len(self.inside_list)
        if self.states[7] == 0:
            if len(self.floor_0_waiting_list) < remaning_places:
                self.inside_list.extend(self.floor_0_waiting_list)
                self.floor_0_waiting_list = []
            else:
                self.inside_list.extend(self.floor_0_waiting_list[:remaning_places])
                self.floor_0_waiting_list = self.floor_0_waiting_list[remaning_places:]
        elif self.states[7] == 1:
            if len(self.floor_1_waiting_list) < remaning_places:
                self.inside_list.extend(self.floor_1_waiting_list)
                self.floor_1_waiting_list = []
            else:
                self.inside_list.extend(self.floor_1_waiting_list[:remaning_places])
                self.floor_1_waiting_list = self.floor_1_waiting_list[remaning_places:]
        elif self.states[7] == 2:
            if len(self.floor_2_waiting_list) < remaning_places:
                self.inside_list.extend(self.floor_2_waiting_list)
                self.floor_2_waiting_list = []
            else:
                self.inside_list.extend(self.floor_2_waiting_list[:remaning_places])
                self.floor_2_waiting_list = self.floor_2_waiting_list[remaning_places:]
        elif self.states[7] == 3:
            if len(self.floor_3_waiting_list) < remaning_places:
                self.inside_list.extend(self.floor_3_waiting_list)
                self.floor_3_waiting_list = []
            else:
                self.inside_list.extend(self.floor_3_waiting_list[:remaning_places])
                self.floor_3_waiting_list = self.floor_3_waiting_list[remaning_places:]
        self.states[6] = len(self.inside_list)

    def step(self, action):
        info = {}
        if self.done:
            print("End of the session")
        else:
            if action == 0:
                self.rotating_people()
                self.time += 10
            if action == 1:
                if self.states[7] == 3:
                    # print("invalid_action")
                    self.reward = self.reward - 1e6
                    self.time+=100
                else:
                    self.time += 2
                    self.states[7] = self.states[7] + 1
            if action == 2:
                if self.states[7] == 0:
                    #print("invalid_action")
                    self.reward = self.reward-1e6
                    self.time += 100
                else:
                    self.time += 2
                    self.states[7] = self.states[7] - 1
            self.reward =self.reward-(self.states[6] + self.states[2] + self.states[1] +
                            self.states[4] + self.states[3] + self.states[0] +
                            self.states[5])
            self.timecheck()
            self.waiting_list_check()
            self.done=self.done_check()

        return self.states,self.reward, self.done, info



#random guess
elevator= Elevator()
elevator.reset()
done=0
while done==0:
    action=np.random.choice(3)
    states,_,done,_=elevator.step(action)
print("random guess elevator time {}".format(elevator.time))

elevator.reset()
model = DQN('MlpPolicy', elevator, verbose=0)
model.learn(total_timesteps=5e4)

