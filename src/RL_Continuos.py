import numpy
import random
import gym
import math
from datetime import datetime

class RL_Continuous:
    def __init__(self):
        self.A = None
        self.n_baseFunctions = 2
    
    def SARSA(self, episode_steps, n_episodes, alfa_in, gamma_in):
        env = gym.make('CartPole-v1')
        self.A = [0,1]
        theta = numpy.random.rand(self.n_baseFunctions,len(self.A))
        alfa = alfa_in
        gamma = gamma_in
        t_total = 0
        numpy.random.seed(1)

        file_log = open('../logs/sarsa'+datetime.now().strftime("%Y%m%d%H%M%S"),'w+')

        epsilon_ep = 0.8
        for i_episode in range(n_episodes):
            st = env.reset()
            st = st[2:]
            t = 0

            at = self.epsilonGreedy(st, theta, epsilon_ep)
            while t < episode_steps:
                #env.render()
                #alfa = alfa*0.999999
                
                st_1, rt, done, info = env.step(at)
                st_1 = st_1[2:]
                at_1 = self.epsilonGreedy(st_1, theta, epsilon_ep)

                c_st_at = self.ValueEstimate(st, theta, at)
                c_st_1_at_1 = self.ValueEstimate(st_1, theta, at_1)
                delta = (rt + gamma * c_st_1_at_1 - c_st_at)
                
                theta[:,at] += alfa * delta * st

                epsilon_ep = epsilon_ep*0.9999999

                st = st_1
                at = at_1              

                t = t+1
                t_total = t_total + 1
                
                if done or t == episode_steps:
                    print("Fim, Tempo:" + str(t))
                    file_log.write(str(i_episode) + ";" + str(t) + "\n")
                    t = episode_steps
                    
        env.close()

    def SARSA2(self, episode_steps, n_episodes, alfa_in, gamma_in):
        env = gym.make('CartPole-v1')
        self.A = [0,1]
        theta = numpy.random.rand(self.n_baseFunctions,len(self.A))
        alfa = alfa_in
        gamma = gamma_in
        t_total = 0
        numpy.random.seed(1)

        file_log = open('../logs/sarsa'+datetime.now().strftime("%Y%m%d%H%M%S"),'w+')

        for i_episode in range(n_episodes):
            st = env.reset()
            st = st[2:]
            t = 0

            at = self.softmax(st, theta)
            while t < episode_steps:
                #env.render()
                #alfa = alfa*0.999999
                
                st_1, rt, done, info = env.step(at)
                st_1 = st_1[2:]
                at_1 = self.softmax(st_1, theta)

                c_st_at = self.ValueEstimate(st, theta, at)
                c_st_1_at_1 = self.ValueEstimate(st_1, theta, at_1)
                delta = (rt + gamma * c_st_1_at_1 - c_st_at)
                
                theta[:,at] += alfa * delta * st

                st = st_1
                at = at_1              

                t = t+1
                t_total = t_total + 1
                
                if done or t == episode_steps:
                    print("Fim, Tempo:" + str(t))
                    file_log.write(str(i_episode) + ";" + str(t) + "\n")
                    t = episode_steps
                    
        env.close()
    
    
    def QLearning(self, episode_steps, n_episodes, alfa_in, gamma_in):
        theta = numpy.zeros(self.n_baseFunctions)
        env = gym.make('CartPole-v1')
        self.A = [0,1]
        alfa = alfa_in
        gamma = gamma_in


        for i_episode in range(n_episodes):
            st = env.reset()
            t = 0
            while t < episode_steps:
                env.render()
                #print(st)

                epsilon_ep = 1/(t+1)
                alfa = alfa*0.98
                at = self.epsilonGreedy(st, theta, epsilon_ep)
                st_1, rt, done, info = env.step(at)

                maxat_1 = self.epsilonGreedy(st_1, theta, 0) #se epslon 0 sempre explota e traz o max

                c_st_at = self.ValueEstimate(st, theta, at)
                c_st_1_at_1 = self.ValueEstimate(st_1, theta, maxat_1)
                delta = (rt + gamma * c_st_1_at_1 - c_st_at)
                
                theta_new = theta + alfa * delta * self.BaseFunctions(st,at)

                theta = theta_new                
                st = st_1                

                t = t+1
                if done or t == episode_steps:
                    print("Fim, Tempo:" + str(t))
                    t = episode_steps
                    
        env.close()
    
    def ValueEstimate(self, obs, theta, at):
        value = numpy.sum(obs * theta[:,at])

        if numpy.isnan(value):
            return 0

        return value

    def epsilonGreedy(self, st, theta, epsilon):
        prob = random.random()
        a = 0
        max_value = 0
        if prob < epsilon:
            a = random.randrange(0,len(self.A))
        else:
            for i in range(len(self.A)):
                aux = self.ValueEstimate(st, theta, i)
                if aux > max_value:
                    max_value = aux
                    a = i
        return a

    def softmax(self, st, theta):
        h = numpy.exp(st.dot(theta))
        h = h / numpy.sum(h)
        return numpy.random.choice(len(self.A),p=h)