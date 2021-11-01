#!/usr/bin/python3

import gym
import minihack

def flatten(x):
    flat_list = [it for sublist in x for it in sublist]
    return flat_list


def combine(x):
    rv = []
    
    for item in x:                
        for i in range(len(item)):
            item[i] = int(item[i])
        rv += item

    return rv

    

# number of states 107 state space (9x9 char surroundings+player stats)
# we use now ints as state passed values
#
# action is number from 0-7 (1-8 values)
# 

env = gym.make("MiniHack-River-v0",
               observation_keys=("chars_crop", "blstats"))

obs = env.reset()

surroundings = flatten(obs["chars_crop"])
stats = obs["blstats"].tolist()
state = combine([surroundings, stats])

done = False


# virtual bool getState(whiteice::math::vertex<T>& state) = 0;

def minihack_getState():
    global env, done, state
    
    if(done == True):
        obs = env.reset()
        
        surroundings = flatten(obs["chars_crop"])
        stats = obs["blstats"].tolist()
        state = combine([surroundings, stats])
        
        done = False
    
    return state


# virtual bool performAction(const whiteice::math::vertex<T>& action,
#                            whiteice::math::vertex<T>& newstate,
#                            T& reinforcement,
#                            bool& endFlag) = 0;

def minihack_performAction(action):
    global env, done, state

    obs, reward, done, info = env.step(action)
    
    surroundings = flatten(obs["chars_crop"])
    stats = obs["blstats"].tolist()
    state = combine([surroundings, stats])

    env.render()
    
    return [state, reward, done]



test_state = minihack_getState()
