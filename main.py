#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:19:32 2023

@author: daniel
"""


import pygame
from pygame.locals import *
import sys
from enviroment import enviroment
import cProfile
import pstats
import os
import numpy as np
import matplotlib.pyplot as plt
from DDQN import DQNagent
import torch 
DEVICE = 'cpu'


def quit_pygame():
    print("Quit")
    pygame.quit()
    sys.exit()
    
def run_manual(params): #Runs the game in manual mode, letting the user control the spacecraft
 
    env = enviroment()
    while True:
        #Quit check
        for event in pygame.event.get():
            if event.type == QUIT:
                quit_pygame()
        #Logic
        action=5 # normally idle
        #Action = 1,2,3,4,5 left,up,right,leftup,rightup,idle
        key = pygame.key.get_pressed()
        if key[pygame.K_UP]: action=1
        if key[pygame.K_RIGHT]: action=2
        if key[pygame.K_LEFT]: action=0
        if key[pygame.K_LEFT] and key[pygame.K_UP]: action=3
        if key[pygame.K_RIGHT] and key[pygame.K_UP]: action=4

        next_state, reward, done, info = env.step(action)
        
        if params['GUI']:
            env.render()
            #env.FramePerSec.tick(env.FPS)
        if done:
            env.reset()
            

def define_parameters():
    params = dict()
    params['replays_per_session']=10
    params['epsilon_max'] = 1.0
    params['epsilon_min'] = 0.01
    params['gamma'] = 0.95
    params['learning_rate'] = 0.001#0.001
    params['first_layer_size'] = 124    # neurons in the first layer
    params['second_layer_size'] = 64   # neurons in the second layer
    params['episodes'] = 20000
    params['memory_size'] = 10000
    params['batch_size'] = 128
    params['weights_path'] = 'weights.txt'
    params['mode'] =0#Mode: 0=train, 1= replay, 2=play manually
    params['epsilon_decay'] =np.e**(np.log(params['epsilon_min'])/(params['episodes']* params['replays_per_session']*0.75))
    #print(params['epsilon_decay'])
    return params  


def run_AI(params):
    
    award_list=[]
    
    env = enviroment()
    
    Q1_agent = DQNagent(params,11,5)
    Q2_agent = DQNagent(params,11,5)
    
    for param in Q2_agent.parameters():
        param.requires_grad = False
    #Transfer weights from Q1 to Q2
    Q1_agent.update_parameters(Q2_agent)
    
    
    
    
    if params['load_weights'] ==True:
        Q1_agent.load_state_dict(torch.load(params['weights_path'])) 
        Q1_agent.update_parameters(Q2_agent)
        Q1_agent.eval()
        Q2_agent.eval()
        
    for i in range(1, params['episodes']):
        score = 0
        for event in pygame.event.get():
            if event.type == QUIT:
                quit_pygame()

        state = env.reset()
        state,_,_,_ = env.step(np.argmax([0,0,0,0,1]))#Start the game and force a first move
        done=False
        while ((not done) and env.score<60):
            for event in pygame.event.get():
                if event.type == QUIT:
                    quit_pygame()
            if params['GUI']:
                env.render()
                #env.FramePerSec.tick(env.FPS)
                
            action = Q2_agent.act(state) #Get best action depening on state 
            frame_counter=0  
            while frame_counter<2: #Frames between sending new moves to the rocket
                next_state, reward, done, info = env.step(action)  
                frame_counter+=1
            reward = reward if not done else -100 # -100 if we crash      
            if params['train']: #Push action and state space to DDQN memory
                Q1_agent.remember(state, action, reward, next_state, done)    
            state = next_state # set state to next state           
            
            score += reward
   
        award_list.append(score)
        #print(score)
        if i%100==0:
            print("Episode {}, Avg Reward {}, Last Reward {}, Epsilon {}".format(i, sum(award_list)/i, award_list[i-1], Q1_agent.returning_epsilon()))
      
        if params['train']:
            for j in range(1, params['replays_per_session']):
                Q1_agent.replay(Q2_agent) #Train using replays from memory 
            #Transfer weights from Q1 to Q2
            Q1_agent.update_parameters(Q2_agent)


    if params['train']:
        model_weights = Q1_agent.state_dict()
        torch.save(model_weights, params["weights_path"])
        print("saved")
    plt.plot(award_list,'*')
    plt.figure()
    plt.plot(np.cumsum(award_list)/range(1,len(award_list)+1))#Incremental mean
    return

if __name__ == '__main__':
    pygame.init()
    pygame.font.init()
    params = define_parameters()
    with cProfile.Profile() as pr:
        if params['mode']==0:#train
            print("Training mode.")
            params['GUI']=False
            params['train']=True
            params['load_weights'] = False   # when training, the network is not pre-trained
            run_AI(params)
        elif params['mode']==1:#replay
            print("Replay mode.")
            params['GUI']=True
            params['train']=False
            params['load_weights'] = True
            run_AI(params)
        elif params['mode']==2:#play
            print("Playing game.")
            params['GUI']=True
            params['train']=False
            params['load_weights'] = False
            run_manual(params) 
        else:
            print("Mode not correct")
            quit_pygame()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats("cprofiler.prof")
    quit_pygame()