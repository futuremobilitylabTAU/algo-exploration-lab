import pandas as pd
import numpy as np
import random
import time


"""
Author:Gabriel Dadashev 4.4.23

This script implements a Markov Decision Process (MDP) using value iteration and policy iteration methods.
The MDP is defined by states, actions, a transition model, and a reward function.

The main class, MarkovDecisionProcess, provides methods for value iteration, policy iteration, and related functions.

The script also includes loading data from CSV files, running the MDP algorithms, and saving results to CSV files.

"""

start_time=time.time()

class MarkovDecisionProcess:
    """A Markov Decision Process, defined by an states, actions, transition model and reward function."""
    def __init__(self, Transitions, Reward, gamma):
        #collect all nodes from the transition models
        self.transition = Transitions
        #initialize reward
        self.reward = Reward
        #initialize gamma
        self.gamma = gamma
        #initialize state
        self.states = self.transition['State'].value_counts().keys().values
        self.best_policy=[]

    def T(self, state, action):
        """for a state and an action, return a list of (probability, next-state) pairs."""
        transition_filter=self.transition.copy()
        transition_filter['pairs']=transition_filter.apply(lambda x:(x['Transitions'],x['Next State']),axis=1)
        transition_filter=transition_filter[(transition_filter['State']==state)&(transition_filter['Action']==action)]
        l=transition_filter.groupby('State')['pairs'].apply(list).to_frame()
        return l.loc[state,'pairs']
    
    def actions(self, state):
        """return set of actions that can be performed in this state"""
        transition_filter=self.transition.copy()
        transition_filter=transition_filter[transition_filter['State']==state]
        transition_filter=transition_filter.groupby(['State','Action'])['Action'].count().to_frame()
        transition_filter=transition_filter.drop('Action',axis=1).reset_index()
        l=transition_filter.groupby('State')['Action'].apply(list).to_frame()
        return l.loc[state,'Action']
    
    def get_best_policy(self):
        return self.best_policy
   
    
    def value_iteration(self):
       """
       Solving the MDP by value iteration.
       returns utility values for states after convergence and optimal policy
       """
       states = self.states
       delta = np.inf
       epsilon =0.1
       V1 = {s: 0 for s in states}
       pi = {s: 0 for s in states}

       count=0
       while delta>epsilon:
           print('iteration',count,delta)
           count=count+1
           delta=0
           V = V1.copy()           
           for s in states:
               maxim=-np.inf
               a_best=None
               for a in self.actions(s):
                   trans=self.T(s,a)
                   summ=0
                   for t in trans:
                       intermediate_reward=float(self.reward[(self.reward['State']==s)&(self.reward['Action']==a)]['Reword'])
                       value=t[0]*(intermediate_reward+self.gamma*V[t[1]])
                       summ=summ+value
                   if summ>maxim: 
                       maxim=summ
                       a_best=a
               pi[s]=a_best
               V1[s]=maxim
               print ("State", s, "V now:",  V1[s],"V old:",  V[s])
               delta=max(delta,abs(V1[s]- V[s]))
       self.best_policy=pi
       return V

    def policy_iteration(self):
        """
        Solving the MDP by policy iteration.
        returns utility values for states after convergence and optimal policy
        """
        states = self.states
        V1 = {s: 0 for s in states}
        pi=self.reward.copy()
        pi=pi.groupby('State')['Action'].apply(list).to_frame()
        pi['Action']= pi['Action'].apply(lambda x:  random.choice(x))
        pi = {s: str(pi.at[s,'Action']) for s in states}
        is_value_changed = True
        iterations = 0
        while is_value_changed:
            print('iteration',iterations)
            is_value_changed = False
            iterations += 1
            for s in states:
                intermediate_reward=float(self.reward[(self.reward['State']==s)&(self.reward['Action']==pi[s])]['Reword'])
                V1[s] = sum([t[0] * (intermediate_reward + gamma*V1[t[1]]) for t in self.T(s,pi[s])])
            
            for s in states:
                q_best = V1[s]
                for a in self.actions(s):
                    intermediate_reward=float(self.reward[(self.reward['State']==s)&(self.reward['Action']==a)]['Reword'])
                    q_sa = sum([t[0] * (intermediate_reward + gamma * V1[t[1]]) for t in self.T(s,a)])
                    if q_sa > q_best:
                        print ("State", s, ": Q now:", q_sa, "Q best:", q_best)
                        pi[s] = a
                        q_best = q_sa
                        is_value_changed = True
                 
        self.best_policy=pi
        return V1



           


# Load data
transitions = pd.read_csv(r'C:\Users\ASUS\Desktop\Thesis_AMOD\assets\control_model\transitions\trans_df.csv')
reward = pd.read_csv(r'C:\Users\ASUS\Desktop\Thesis_AMOD\assets\control_model\reword\rew_df.csv')
gamma = 0.9

# Create MDP instance
mdp = MarkovDecisionProcess(transitions, reward, gamma)

# Run value iteration  / policy iteration
v1 = mdp.value_iteration()
elapsed_time = time.time() - start_time
print(f"Elapsed Time: {elapsed_time} seconds\n")

# Get the best policy
best_policy = mdp.get_best_policy()

# Print the final state values and optimal policy
print('Final state-values:', v1)
print('Optimal policy:', best_policy)

# Convert policy and state values to DataFrames for easier handling
pi_df = pd.DataFrame(list(v1.items()), columns=['State', 'Action'])
v_df = pd.DataFrame(list(best_policy.items()), columns=['State', 'Value'])

# Save DataFrames to CSV files
pi_df.to_csv(r"C:\Users\ASUS\Desktop\Thesis_AMOD\assets\optimal_policy\optimal_VI.csv", index=False)
v_df.to_csv(r"C:\Users\ASUS\Desktop\Thesis_AMOD\assets\optimal_policy\optimal_state_value.csv", index=False)

