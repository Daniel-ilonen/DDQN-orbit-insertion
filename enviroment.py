#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 02:34:56 2023

@author: daniel
"""
import pygame
from player import Player
import math
from planet import Planet

class enviroment:

    def __init__(self):
        display_screen = pygame.display.set_mode((800,600))# Setting a random caption title for your pygame graphical window.
        pygame.display.set_caption('DDQN rocketship')  
        self.score = 0
        self.game_status=0
        self.FPS=30
        self.FramePerSec = pygame.time.Clock()
        #self.crash=0
        self.display_screen=display_screen
        self.Earth=Planet((300,300))
        self.P1 = Player()
        self.P1.update(self.Earth)
        
        self.reward = 0
    def reset(self):
        
        self.score = 0
        self.game_status=0
        
        #self.crash=0
       
        self.Earth=Planet((300,300))
        self.P1 = Player()
        self.P1.update(self.Earth)
     
        self.reward = 0
        state_init = self.get_state() 

        return state_init
    def step(self,action):
#        print(action)
        self.P1.queueMove_AI(action)

        if pygame.sprite.collide_circle(self.P1, self.Earth):
            self.P1.crash()
           
        else:
           
            self.P1.update(self.Earth)
            if self.P1.docked==0:
                self.score+= 1/30
        
        done= self.P1.crashed          
        reward=self.set_reward()
        next_state=self.get_state()
        info=0
        return next_state, reward, done, info
                    
    def get_state(self):
        
        state = [self.P1.angular_velocity, #-3 - 3
                 2*math.sqrt(self.P1.v2), #0-1
                 self.P1.angle*math.pi/180,
                 self.P1.theta_prograde, #0-2pi
                 self.P1.theta_vel, #0-2pi
                 math.log(self.P1.height)-5, #0-2
                 self.P1.fuel/100, #0-1
                 self.P1.rho,
                 (self.P1.rho<0.01)*1,#0-1
                 self.P1.theta_grav,
                 self.P1.theta_traveled#0-2pi
                 #self.score
                 ]

        return state

    def set_reward(self):
        self.reward =(1-self.P1.rho)*5*self.P1.theta_traveled
        return self.reward   
    
    def render(self):
        self.display_screen.fill((255, 255, 255)) 
        self.Earth.draw(self.display_screen,self.Earth.ratio_atmos_earth)
        self.P1.draw(self.display_screen,self.Earth.CoG)
        self.P1.draw_flame(self.display_screen)
        pygame.draw.line(self.display_screen, (0,0,255), (700,500),(700,500-self.P1.fuel), 20)

        font = pygame.font.SysFont( None, 20 )
        number_image_timer = font.render(str(round(self.score,1)), True, (0,0,0))
        number_image_fuel = font.render(str(round(self.P1.fuel,0)), True, (0,0,0))
       
        self.display_screen.blit(number_image_timer,(300,10))
        self.display_screen.blit(number_image_fuel,(690,500))
        pygame.display.update()
        