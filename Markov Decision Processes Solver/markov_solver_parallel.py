"""
Markov Decision Process (MDP) Solver 

This script defines a Markov Decision Process (MDP) and employs parallel value iteration
to find the optimal policy and state values. The MDP is characterized by states, actions,
a transition model, and a reward function. The script utilizes multiprocessing to parallelize
the computation across multiple worker processes, enhancing performance on multi-core systems.
17.1.24
Author: Gabi Dadashev 
"""

from time import sleep
from random import random
from multiprocessing import Process, Manager
import multiprocessing
import pandas as pd
import numpy as np
import random
import math
import threading
import time
import os


class MarkovDecisionProcess:
    """A Markov Decision Process, defined by states, actions, transition model, and reward function."""

    def __init__(self, Transitions, Reward, gamma):
        """
        Initialize MarkovDecisionProcess with transition model, reward function, and discount factor.

        Parameters:
        - Transitions: DataFrame, transition model
        - Reward: DataFrame, reward function
        - gamma: float, discount factor
        """
        self.transition = Transitions
        self.reward = Reward
        self.gamma = gamma
        self.states = self.transition['State'].value_counts().keys().values
        self.best_policy = []
        self.V1 = {s: 0 for s in self.states}
        self.pi = {s: 0 for s in self.states}
        self.delta = np.inf

    def T(self, state, action):
        """For a state and an action, return a list of (probability, next-state) pairs."""
        transition_filter = self.transition.copy()
        transition_filter['pairs'] = transition_filter.apply(lambda x: (x['Transitions'], x['Next State']), axis=1)
        transition_filter = transition_filter[(transition_filter['State'] == state) & (transition_filter['Action'] == action)]
        l = transition_filter.groupby('State')['pairs'].apply(list).to_frame()
        return l.loc[state, 'pairs']

    def actions(self, state):
        """Return a set of actions that can be performed in this state."""
        transition_filter = self.transition.copy()
        transition_filter = transition_filter[transition_filter['State'] == state]
        transition_filter = transition_filter.groupby(['State', 'Action'])['Action'].count().to_frame()
        transition_filter = transition_filter.drop('Action', axis=1).reset_index()
        l = transition_filter.groupby('State')['Action'].apply(list).to_frame()
        return l.loc[state, 'Action']

    def get_best_policy(self):
        """Return the best policy."""
        return self.best_policy


def worker_function(worker_id, m, shared_dict, states, shared_policy, shared_delta):
    """
    Perform value iteration for a subset of states.

    Parameters:
    - worker_id: int, identifier for the worker process
    - m: MarkovDecisionProcess, the Markov Decision Process
    - shared_dict: Manager.dict, shared state-value dictionary
    - states: list, subset of states for this worker
    - shared_policy: Manager.dict, shared policy dictionary
    - shared_delta: Manager.dict, shared delta values for convergence
    """
    local_delta = 0

    for s in states:
        maxim = -np.inf
        a_best = None

        for a in m.actions(s):
            trans = m.T(s, a)
            summ = 0

            for t in trans:
                intermediate_reward = float(m.reward[(m.reward['State'] == s) & (m.reward['Action'] == a)]['Reword'])
                value = t[0] * (intermediate_reward + m.gamma * shared_dict[t[1]])
                summ += value

            if summ > maxim:
                maxim = summ
                a_best = a

        shared_policy[s] = a_best
        local_delta = max(local_delta, abs(maxim - shared_dict[s]))
        print("State", s, "V now:", maxim, "V old:", shared_dict[s])
        shared_dict[s] = maxim

    shared_delta[worker_id] = local_delta


def parallel_iteration(worker_function, m, shared_dict, shared_policy, shared_delta, num_workers, states):
    """
    Perform parallel value iteration using multiple worker processes.

    Parameters:
    - worker_function: function, the worker function to be executed by each process
    - m: MarkovDecisionProcess, the Markov Decision Process
    - shared_dict: Manager.dict, shared state-value dictionary
    - shared_policy: Manager.dict, shared policy dictionary
    - shared_delta: Manager.dict, shared delta values for convergence
    - num_workers: int, number of worker processes
    - states: list of lists, partitioned states for each worker
    """
    processes = []

    for i in range(num_workers):
        process = Process(target=worker_function, args=(i, m, shared_dict, states[i], shared_policy, shared_delta))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


def main():
    start_time = time.time()
    print("Number of CPU cores:", os.cpu_count())

    # Load data generated by MDP Creation Algorithm
    Transitions = pd.read_csv(r'C:\Users\ASUS\Desktop\Thesis_AMOD\assets\control_model\transitions\trans_df.csv')
    Reward = pd.read_csv(r'C:\Users\ASUS\Desktop\Thesis_AMOD\assets\control_model\reword\rew_df.csv')
    gamma = 0.9

    m = MarkovDecisionProcess(Transitions, Reward, gamma)

    states = m.states
    num_workers = 10
    lists = [states[math.ceil(i): math.ceil(i + len(states) / num_workers)] for i in
             (len(states) / num_workers * j for j in range(num_workers))]

    epsilon = 0.1
    count = 0

    with Manager() as manager:
        # Create shared dictionaries
        shared_dict = manager.dict(m.V1)
        shared_policy = manager.dict()
        shared_delta = manager.dict()

        while m.delta > epsilon:
            print('iteration', count, m.delta)
            count += 1
            m.delta = 0

            parallel_iteration(worker_function, m, shared_dict, shared_policy, shared_delta, num_workers, lists)

            m.delta = max(shared_delta.values())

            # Update m.V1 with a copy of shared_dict
            m.V1 = dict(shared_dict)
            m.pi = dict(shared_policy)

    elapsed_time = time.time() - start_time
    print(f"Elapsed Time: {elapsed_time} seconds\n")
    return (m)
    
    


if __name__ == "__main__":
    # Execute the main function and retrieve the Markov Decision Process instance
    m = main()

    # Print the final state values and optimal policy
    print('Final state-values:', m.V1)
    print('Optimal policy:', m.pi)

    # Convert policy and state values to DataFrames for easier handling
    pi_df = pd.DataFrame(list(m.pi.items()), columns=['State', 'Action'])
    v_df = pd.DataFrame(list(m.V1.items()), columns=['State', 'Value'])

    # Save DataFrames to CSV files
    pi_df.to_csv(r"C:\Users\ASUS\Desktop\Thesis_AMOD\assets\optimal_policy\optimal_VI.csv", index=False)
    v_df.to_csv(r"C:\Users\ASUS\Desktop\Thesis_AMOD\assets\optimal_policy\optimal_state_value.csv", index=False)


