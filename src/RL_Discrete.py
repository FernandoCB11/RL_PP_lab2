import numpy
import random
import gym
import math
from datetime import datetime

class RL_Discrete:
    def __init__(self):
        self.A = None
        self.n_baseFunctions = 2
        self.Q = None
        self.QHits = None

    def SARSA(self, episode_steps, n_episodes, alfa_in, gamma_in):
        env = gym.make('CartPole-v1')
        self.A = [0,1]
        self.Q = numpy.zeros((120,2))
        self.QHits = numpy.zeros((120,2))
        alfa = alfa_in
        gamma = gamma_in
        t_total = 0
        numpy.random.seed(1)

        file_log = open('../logs/sarsa'+datetime.now().strftime("%Y%m%d%H%M%S"),'w+')

        epsilon_ep = 1.0
        for i_episode in range(n_episodes):
            st = env.reset()
            t = 0
            
            at = self.epsilonGreedy(st, epsilon_ep)
            while t < episode_steps:
                #env.render()
                alfa = alfa*0.999
                base = self.BaseFunctions(st)

                st_1, rt, done, info = env.step(at)
                at_1 = self.epsilonGreedy(st_1, epsilon_ep)

                if done:
                    rt = 0

                c_st_at = self.ValueEstimate(st, at)
                c_st_1_at_1 = self.ValueEstimate(st_1, at_1)
                delta = (rt + gamma * c_st_1_at_1 - c_st_at)
                
                self.updateValues(st,at,alfa,delta)
                epsilon_ep = epsilon_ep*0.999

                #print(st[3])
                
                st = st_1
                at = at_1              

                t = t+1
                t_total = t_total + 1
                
                if done or t == episode_steps:
                    print("Fim, Tempo:" + str(t))
                    file_log.write(str(i_episode) + ";" + str(t) + "\n")
                    t = episode_steps
        
        env.close()
        
        return numpy.argmax(self.Q,axis=1)


    def REINFORCE(self, episode_steps, n_episodes, alfa_in, gamma_in):
        env = gym.make('CartPole-v1')
        theta = numpy.random.rand(4, 2)
        self.A = [0,1]
        alfa = alfa_in
        gamma = gamma_in
        t_total = 0
        numpy.random.seed(1)

        file_log = open('../logs/reinforce'+datetime.now().strftime("%Y%m%d%H%M%S"),'w+')

        for i_episode in range(n_episodes):
            st = env.reset()[None,:]
            t = 0
            theta_new = theta
            gradients = []
            rs = []
            while t < episode_steps:
                #env.render()
                #alfa = alfa*0.999999

                at = self.chooseAction(st, theta)

                st_1, rt, done, info = env.step(at)
                st_1 = st_1[None,:]

                gradients.append(self.findGradient(st, theta, at))
                rs.append(rt)

                st = st_1

                t = t+1
                t_total = t_total + 1
                
                if done or t == episode_steps:
                    for i in range(len(gradients)):
                        theta_new += alfa * gradients[i] * sum([ r * (gamma ** t) for t,r in enumerate(rs[i:])])
                    print("Fim, Tempo:" + str(t))
                    file_log.write(str(i_episode) + ";" + str(t) + "\n")
                    t = episode_steps
            
            theta = theta_new

        return None

    def findGradient(self, st,theta,at):
        pi = (self.paramPolicy(st, theta))
        pi_r = pi.reshape(-1,1)
        aux = numpy.diagflat(pi_r) - numpy.dot(pi_r, pi_r.T)
        aux = aux[at,:]

        dlog = aux / pi[0,at]
        gradient = st.T.dot(dlog[None,:])

        return gradient        

    def chooseAction(self, st, theta):
        pi = self.paramPolicy(st, theta)
        return numpy.random.choice(len(self.A),p=pi[0])
    
    def paramPolicy(self, st, theta):
        h = numpy.exp(st.dot(theta))
        return h / numpy.sum(h)
    
    def BaseFunctions2(self, obs):
        ret = numpy.zeros(2, dtype=int)

        if obs[2] <= -0.20944:
            ret[0] = -5
        elif obs[2] <= -0.139626:
            ret[0] = -4
        elif obs[2] <= -0.10472:
            ret[0] = -3
        elif obs[2] <= -0.069813:
            ret[0] = -2
        elif obs[2] <= -0.034907:
            ret[0] = -1
        elif obs[2] <= 0:
            ret[0] = 0
        elif obs[2] <= 0.034907:
            ret[0] = 1
        elif obs[2] <= 0.069813:
            ret[0] = 2
        elif obs[2] <= 0.10472:
            ret[0] = 3
        elif obs[2] <= 0.139626:
            ret[0] = 4
        elif obs[2] <= 0.20944:
            ret[0] = 5
        else:
            ret[0] = 6

        if obs[3] < -2.0:
            ret[1] = -4
        elif obs[3] < -1.5:
            ret[1] = -3
        elif obs[3] < -1.0:
            ret[1] = -2
        elif obs[3] < -0.5:
            ret[1] = -1
        elif obs[3] < 0:
            ret[1] = 0
        elif obs[3] < 0.5:
            ret[1] = 1
        elif obs[3] < 1.0:
            ret[1] = 2
        elif obs[3] < 1.5:
            ret[1] = 3
        elif obs[3] < 2.0:
            ret[1] = 4
        else:
            ret[1] = 5

        return ret
    
    def QLearning(self, episode_steps, n_episodes, alfa_in, gamma_in):
        env = gym.make('CartPole-v1')
        self.A = [0,1]
        self.Q = numpy.zeros((120,2))
        alfa = alfa_in
        gamma = gamma_in
        t_total = 0

        for i_episode in range(n_episodes):
            st = env.reset()
            t = 0
            while t < episode_steps:
                env.render()
                #print(st)

                epsilon_ep = 1/(t+1)
                alfa = alfa*0.9
                at = self.epsilonGreedy(st, epsilon_ep)
                st_1, rt, done, info = env.step(at)

                maxat_1 = self.epsilonGreedy(st_1, 0) #se epslon 0 sempre explota e traz o max

                c_st_at = self.ValueEstimate(st, at)
                c_st_1_at_1 = self.ValueEstimate(st_1, maxat_1)
                delta = (rt + gamma * c_st_1_at_1 - c_st_at)
                
                self.updateValues(st,at,alfa,delta)

                st = st_1                

                t = t+1
                t_total = t_total + 1
                if done or t == episode_steps:
                    print("Fim, Tempo:" + str(t))
                    t = episode_steps
                    
        env.close()
    
    def ValueEstimate(self, obs, at):
        value = 0
        base = self.BaseFunctions(obs)

        value = self.Q[base[0],at]
        
        return value

    def BaseFunctions(self, obs):
        ret = numpy.zeros(1, dtype=int)

        #ret[0] = int(math.floor((obs[0]*10 + 48) / 12) - 1)

        #if obs[0] <= -2.4:
        #    ret[0] = 0
        #elif obs[0] <= -1.4:
        #    ret[0] = 1
        #elif obs[0] <= -0.4:
        #    ret[0] = 2
        #elif obs[0] <= 0.0:
        #    ret[0] = 3
        #elif obs[0] <= 0.4:
        #    ret[0] = 4
        #elif obs[0] <= 1.4:
        #    ret[0] = 5
        #elif obs[0] <= 2.4:
        #    ret[0] = 6
        #else:
        #    ret[0] = 7
        
        #if obs[1] < 0:
        #    ret[1] = 0
        #else:
        #    ret[1] = 1

        if obs[2] <= -0.20944:
            ret[0] = 0
        elif obs[2] <= -0.139626:
            ret[0] = 1
        elif obs[2] <= -0.10472:
            ret[0] = 2
        elif obs[2] <= -0.069813:
            ret[0] = 3
        elif obs[2] <= -0.034907:
            ret[0] = 4
        elif obs[2] <= 0:
            ret[0] = 5
        elif obs[2] <= 0.034907:
            ret[0] = 6
        elif obs[2] <= 0.069813:
            ret[0] = 7
        elif obs[2] <= 0.10472:
            ret[0] = 8
        elif obs[2] <= 0.139626:
            ret[0] = 9
        elif obs[2] <= 0.20944:
            ret[0] = 10
        else:
            ret[0] = 11

        #ret[2] = int(math.floor((obs[2] + 24) / 6) - 1)
        
        if obs[3] < -2.0:
            ret[0]
        elif obs[3] < -1.5:
            ret[0] = ret[0] + 12
        elif obs[3] < -1.0:
            ret[0] = ret[0] + 24
        elif obs[3] < -0.5:
            ret[0] = ret[0] + 36
        elif obs[3] < 0:
            ret[0] = ret[0] + 48
        elif obs[3] < 0.5:
            ret[0] = ret[0] + 60
        elif obs[3] < 1.0:
            ret[0] = ret[0] + 72
        elif obs[3] < 1.5:
            ret[0] = ret[0] + 84
        elif obs[3] < 2.0:
            ret[0] = ret[0] + 96
        else:
            ret[0] = ret[0] + 108

        return ret

    def epsilonGreedy(self, st, epsilon):
        prob = random.random()
        a = 0
        max_value = 0
        if prob < epsilon:
            a = numpy.random.choice(len(self.A),p=[0.5,0.5])
        else:
            for i in range(len(self.A)):
                aux = self.ValueEstimate(st, i)
                if aux >= max_value:
                    max_value = aux
                    a = i
        return a

    def updateValues(self, obs, at, alfa, delta):
        base = self.BaseFunctions(obs)
        self.Q[base[0],at] = self.Q[base[0],at] + alfa * delta
        self.QHits[base[0],at] += 1
