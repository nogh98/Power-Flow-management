# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 17:30:23 2021

@author: NOGH98
"""
# import the useful modules
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
import time
from utils.calculate_reward import calculate_reward
from utils.random_action import random_action
#ignoring the warnings:
warnings.filterwarnings("ignore")

# Loading the data
data = pd.read_csv('Data2.csv')

# seperating each column of the data 
H = data['H']
Load = data['LOAD']
PV = data['PV']
Tarif = data['Tarif']

# avoiding very high & very low soc level:
Eb_minimum = 10
Eb_maximum = 90

# initializing our decisions:
Eb = np.zeros((25,))
Eb = Eb.tolist()
Eb2 = np.zeros((25,))
Eb2 = Eb2.tolist()
Eb[0] = 50
Eb2[0] = 50
power_deficit = Load - PV










































lr = 0.001
# epsilon = 1.
gamma = 1.
# eps_decay = .9


# avoiding high - low soc level:
Eb_minimum = 10
Eb_maximum = 90

# exploration - exploitation dilemma
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay = 0.01

Pg = np.arange(-5000, 9001, 1)
num_epochs = 14500

Q_dict = {}

for epoch in range(num_epochs):
#     print('at epoch: ', epoch)

    for h in range(int(max(H)) + 1):
        
        #print('we are at time ', h)
        incomplete_state = [str(h), str(PV[h]), str(Load[h]), str(Tarif[h])]
        incomplete_state.append(str(Eb[h]))
        state_string = '_'.join(incomplete_state)
        #print('state string is: ', state_string)
        if state_string not in list(Q_dict.keys()):
            
            #print(state_string, ' has not been visited before.')
            Dict = {}

            Pbss = np.arange(100*(Eb[h] - Eb_maximum), 100*(Eb[h] - Eb_minimum), 1)
            #print('Pbss is from %.5f to %.5f'%(min(Pbss), max(Pbss)))
            all_decisions = np.array(np.meshgrid(np.around(Pg/100, decimals = 2),
                                                 np.around(Pbss/100, decimals = 2))).T.reshape(-1, 2)

#             Sum_decisions = np.around(np.sum(all_decisions, axis = 1), decimals = 2)
            #print('all decisions before sum: ', len(all_decisions))

    
            all_decisions = all_decisions[
                np.sum(all_decisions, axis = 1) == power_deficit[h]
            ]

            #print('all decision after sum: ', len(all_decisions))
            #print('we found %d possible actions for state %s'%(len(all_decisions), state_string))
            for Pg1, Pbss1 in all_decisions:
                #if h == 0:
                    #print('Pg1 is: ', Pg1, 'Pbss1 is: ', Pbss1)
                action_vector = [str(Pg1), str(Pbss1)]
                action_identity = '_'.join(action_vector)
                Dict[action_identity] = 0
            Q_dict[state_string] = Dict
            all_decisions, Pbss, Dict = None, None, None

        #else:
            #print('we have seen %s before.'%(state_string))

        if np.random.uniform() < epsilon:
            
            action_string = random_action(Q_dict[state_string])
            #print('we selected a random action %s with proba: %.3f'%(action_string, proba))
            #print('we have selected: ', action_string)
        else:
            
            action_string = list(Q_dict[state_string].keys())[list(Q_dict[state_string].values()).index(max(Q_dict[state_string].values()))]
            #print('we selected greedy action: ', action_string)

        selected_Pg, selected_Pbss = action_string.split('_')
        selected_Pg, selected_Pbss = float(selected_Pg), float(selected_Pbss)
        Eb[h + 1] = np.around(Eb[h] - selected_Pbss, decimals = 2)
        reward = calculate_reward(selected_Pbss, selected_Pg, Eb2 = Eb[h + 1], Eb1 = Eb[h], Tarif = Tarif[h])
        #print('with Pbss = %.5f and Pg = %.5f, reward is %.5f'%(selected_Pbss, selected_Pg, reward))
        if h != 23:


            next_state_string = [str(h+1), str(PV[h+1]), str(Load[h+1]), str(Tarif[h+1]), str(Eb[h+1])]
            next_state_string = '_'.join(next_state_string)
            #print('next_state_string is: ', next_state_string)

            if next_state_string not in list(Q_dict.keys()):
                Dict = {}

                Pbss = np.arange(100*(Eb[h + 1] - Eb_maximum), 100*(Eb[h + 1] - Eb_minimum), 1)
                
                all_decisions = np.array(np.meshgrid(np.around(Pg/100, decimals = 2), np.around(Pbss/100, decimals = 2))).T.reshape(-1, 2)
#                 Sum_decisions = np.around(np.sum(all_decisions, axis = 1), decimals = 2)

                all_decisions = all_decisions[
                    np.around(np.sum(all_decisions, axis = 1), decimals = 2) == np.around(power_deficit[h+1], decimals = 2)
                ]
                #print('we found %d possible actions for this next state'%(len(all_decisions)))
                for Pg2, Pbss2 in all_decisions:
                            action_vector = [str(Pg2), str(Pbss2)]
                            action_identity = '_'.join(action_vector)
                            Dict[action_identity] = 0
                Q_dict[next_state_string] = Dict
                all_decisions, Pbss, Dict = None, None, None
            #else: continue; print(next_state_string, 'is already visited')
            Q_dict[state_string][action_string] += lr * (reward + gamma * max(Q_dict[next_state_string].values()) - Q_dict[state_string][action_string])
        else: Q_dict[state_string][action_string] += lr * (reward - Q_dict[state_string][action_string])
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * epoch)
    if epoch % 100 == 0:
        print('for epoch ',epoch + 1,' epsilon is: ', epsilon)
    if epoch >= 12000:
        print(epoch)



# policy retrieval:
Eb_final = np.zeros((25,))
Pbss_final = np.zeros_like(Eb_final)
Pg_final = np.zeros_like(Pbss_final)

Eb_final = Eb_final.tolist()
Pbss_final = Pbss_final.tolist()
Pg_final = Pg_final.tolist()



states = list(Q_dict.keys())
current_state = states[0]
for h in range(24):
    print('current state is: ', current_state)
    h2, pv, L, T, E = current_state.split('_')
    
    Eb_final[h] = E
    action_string = list(Q_dict[current_state].keys())[list(Q_dict[current_state].values()).index(max(Q_dict[current_state].values()))]
    print('action string is: ', action_string)
    Pg, Pbss = action_string.split('_')
    
    Pbss_final[h] = float(Pbss)
    Pg_final[h] = float(Pg)
    E2 = np.around(float(E) - float(Pbss), decimals = 2)
    current_state_list = [str(h + 1), str(PV[h+1]), str(Load[h+1]), str(Tarif[h+1]), str(E2)]
    current_state = '_'.join(current_state_list)
