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

def convert_chars(ls):
    for i in range(len(ls)):
        if(ls[i] == 32): # stone
            ls[i] = 10
        elif(ls[i] == 46): # open floor
            ls[i] = 0
        elif(ls[i] == 64): # @ character
            ls[i] = 1
        elif(ls[i] == '<'):
            ls[i] = 5
        elif(ls[i] == '>'):
            ls[i] = 6
        else:
            ls[i] = 5
            
    return ls

# MiniHack-MazeWalk-45x19-v0


env = gym.make("MiniHack-MazeWalk-45x19-v0",
               observation_keys=("chars_crop", "blstats"))

#env = gym.make("MiniHack-River-v0",
#               observation_keys=("chars_crop", "blstats"))

obs = env.reset()

env.render()

surroundings = convert_chars(flatten(obs["chars_crop"]))

print(surroundings)


