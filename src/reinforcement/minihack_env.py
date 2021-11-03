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

env_name = "MiniHack-River-v0"
env_name = "MiniHack-MazeWalk-45x19-v0"
env_name = "MiniHack-MazeWalk-Mapped-15x15-v0"

env = gym.make(env_name, observation_keys=("chars_crop", "blstats"))

render = False

#########################################################

obs = env.reset()
moves = 0

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
    global env, done, state,moves

    obs, reward, done, info = env.step(action)
    
    surroundings = flatten(obs["chars_crop"])
    stats = obs["blstats"].tolist()
    state = combine([surroundings, stats])

    if(render):
        print("moves: " + str(moves) + " reward = " + str(reward))
        env.render()
    
    moves = moves + 1
    
    return [state, reward, done]



test_state = minihack_getState()
