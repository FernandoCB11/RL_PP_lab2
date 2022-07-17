from solvers import LSTDQSolver
from policy import Policy
from basis_functions import RadialBasisFunction
from basis_functions import OneDimensionalPolynomialBasis
from basis_functions import MultiDimensionalPolynomialBasis
from lspi import learn
from sample import Sample
import numpy
import gym
from datetime import datetime

solver = LSTDQSolver(precondition_value=0.0)
#basis = RadialBasisFunction(numpy.array([[0,0,0,0], [2,2,2,2], [4,4,4,4], [6,6,6,6]]),gamma=0.5,num_actions=2)
#basis = RadialBasisFunction(means=numpy.array([[0,0,0,0],[1.0,0.5,0.05,0.5],[2.0,1.0,0.1,1.0],[1.5,1.5,0.2,1.5]]),gamma=0.0001,num_actions=2)
#basis = OneDimensionalPolynomialBasis(2,2)
basis = MultiDimensionalPolynomialBasis(4,2)
pol = Policy(basis=basis,discount=0.999999,explore=0.5,weights=None,tie_breaking_strategy=Policy.TieBreakingStrategy.RandomWins)
rollout = 500
trains = 4
test = 1000

env = gym.make('CartPole-v1')

file_log = open('../logs/LSPI'+datetime.now().strftime("%Y%m%d%H%M%S"),'w+')

data = []
for k in range(trains):
    #data = []
    for i in range(rollout):
        st = env.reset()
        #st = numpy.array([st[2]])
        done = False
        while done == False:
            at = pol.select_action(st)
            st_1, rt, done, info = env.step(at)
            #st_1 = numpy.array([st_1[2]])
            s = Sample(state=st,action=at,reward=rt,next_state=st_1,absorb=done)
            #s = Sample(state=st,action=at,reward=rt,next_state=st_1)
            data.append(s)
            st = st_1

    new_pol = learn(data=data,initial_policy=pol,solver=solver,epsilon=10**-5,max_iterations=10000)

    pol = new_pol
    pol.explore = pol.explore/2

    for j in range(test):
        st = env.reset()
        done = False
        t=0
        while done == False:
            at = new_pol.best_action(st)
            st_1, rt, done, info = env.step(at)
            st = st_1
            t = t+1
        
        print("Fim, Tempo:"+ str(k) + ";" + str(j) + ";" + str(t))
        file_log.write(str(k) + ";" + str(j) + ";" + str(t) + "\n")

print("Fim")