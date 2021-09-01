# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 17:03:41 2021

@author: NOGH98
"""
import numpy as np
def random_action(dict):
        actions = list(dict.keys())
        return np.random.choice(actions)