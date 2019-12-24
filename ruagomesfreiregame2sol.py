#Grupo 35: Filipe Reynaud N.86412, David Cruz N.89377
import random
import numpy as np

# LearningAgent implemented
class LearningAgent:

        # init
        # nS maximum number of states
        # nA maximum number of action per state
        def __init__(self,nS,nA):

                self.nS = nS
                self.nA = nA

                # Initialize q-table values to 1
                self.matrizQs = np.ones((nS-1, nA))
                # Set the percent to explore
                self.epsilon = 0.5
                # Set the discount
                self.gamma = 0.9
                # Set the learning rate
                self.alpha = 0.5



        # Select one action, used when learning  
        # st - is the current state        
        # aa - is the set of possible actions
        # for a given state they are always given in the same order
        # returns
        # a - the index to the action in aa
        def selectactiontolearn(self,st,aa):

                if random.uniform(0, 1) < self.epsilon:
                        # Exploration
                        a = aa.index(random.choice(aa))
                else:
                        # Exploitation
                        q_values = []
                        for action in range(len(aa)):
                                q_values.append(self.matrizQs[st-1, action])

                        Qmax = max(q_values)
                        
                        indexes = []
                        for i in range(len(q_values)):
                                if(q_values[i] == Qmax):
                                        indexes.append(i)
                        
                        a = random.choice(indexes)

                return a

        # Select one action, used when evaluating
        # st - is the current state        
        # aa - is the set of possible actions
        # for a given state they are always given in the same order
        # returns
        # a - the index to the action in aa
        def selectactiontoexecute(self,st,aa):
                
                q_values = []
                for action in range(len(aa)):
                        q_values.append(self.matrizQs[st-1, action])

                Qmax = max(q_values)
                
                indexes = []
                for i in range(len(q_values)):
                        if(q_values[i] == Qmax):
                                indexes.append(i)
                
                a = random.choice(indexes)

                return a


        # this function is called after every action
        # st - original state
        # nst - next state
        # a - the index to the action taken
        # r - reward obtained
        def learn(self,ost,nst,a,r):
                '''
                In the beginning of the learning process, all the values of the Q-table are ones, so
                the Qmax when first visiting a state will always be equal to one. As time goes by,
                the values in the Q-table are updated. So, to ensure that we just consider the values
                that were updated (and not the ones that are never updated, which corresponds to the 
                actions that the current state cannot execute), we insert them into an array. With this,
                when we search for the maximum value we get a value that corresponds to a possible action
                that can be executed (we're not going to get the default value, which is one). 
                '''

                q_line = self.matrizQs[nst-1, :]
                values_to_be_considered = []
                for i in q_line:
                        if i != 1:
                                values_to_be_considered.append(i)
                
                if(len(values_to_be_considered) != 0):
                        Qmax = np.max(values_to_be_considered)
                
                else:
                        Qmax = 1

                b = self.matrizQs[ost-1, a]
                self.matrizQs[ost-1, a] = b + self.alpha * (r + self.gamma * Qmax - b)
