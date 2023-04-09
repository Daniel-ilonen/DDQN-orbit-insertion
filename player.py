#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:35:40 2023

@author: daniel
"""
import pygame 
import math
import numpy as np

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__() 
        self.og_image =pygame.image.load("img/Player.png")
        self.og_image =pygame.transform.scale(self.og_image, (10,20))
        self.og_flame_sprite = pygame.image.load("img/flame.png") 
        self.og_flame_sprite =pygame.transform.scale(self.og_flame_sprite, (10,20))
        
        self.flame=self.og_flame_sprite
        self.image = self.og_image
        self.rect = self.image.get_rect()
        
        self.radius=5
        self.velocity=(0.0,0.0)
        self.F_gravity=(0.0,0.0)
        self.F_rocket=(0.0,0.0)
        self.p_rocket=0.0112
        self.position=(300.0,300.0-100-10)
        self.angle = 0
        self.angular_velocity=0
        self.old_positions =[]
        self.fps_counter=0
        self.fuel=100
        self.firing=False
        self.docked=1
        self.movestatus=[0,0,0,0]#left, up, right, idle
        self.crashed=False
        self.v2=0
        self.theta_vel=0
        self.theta_prograde=0
        self.grade_vector=(1,1)
        self.height=100
        self.rho=0
        self.theta_grav=0
        self.theta_traveled=0
        #print("reset")
    def rot(self):
        self.image = pygame.transform.rotate(self.og_image, self.angle)
        self.flame = pygame.transform.rotate(self.og_flame_sprite, self.angle)
        self.angle += self.angular_velocity
        self.angle = self.angle % 360
        self.rect = self.image.get_rect(center=self.rect.center)    
        

          
    def queueMove_AI(self,move,): 
        """
        The DDQN saves actions states as the following ints:
            0 = rotate left
            1 = fire rocket
            2 = rotate right
            3 = rotate left while firing rocket
            4 = rotate right while firing rocket
            5 = idle (not used by DDQN)
        We use this function to convert this int to a more usable array of 0s and 1s used by the game:
            [left, up, right, idle]
        """
        self.movestatus=[move==0 or move==3,
                         move==1 or move==3 or move==4,
                         move==2 or move==4,
                         move==5
                         ]  

    def norm(self,v):
        v_norm=math.sqrt(v[0]*v[0]+v[1]*v[1]) 
        if v_norm==0:
            print("ERRRO")
        return np.float64(v_norm)  
    def dot(self,v1,v2):
        vdot=v1[0]*v2[0]+v1[1]*v2[1]
        return vdot  
    def clamp(self,n, minn, maxn):
        return max(min(maxn, n), minn)
    def update(self,earth):
        
        CoG=earth.CoG
        g=earth.g
       
        self.height=math.sqrt((self.position[1]-CoG[1])**2+(self.position[0]-CoG[0])**2)
        self.F_rocket=(0.0,0.0)
        self.F_gravity=(0.0,0.0)  
        rho=earth.get_rho(self.height)
        self.rho=rho
       
        #Gravity
        alpha=math.atan2(-(self.position[1]-CoG[1]),-(self.position[0]-CoG[0]))
        self.F_gravity=(g*math.cos(alpha),g*math.sin(alpha))
        
        self.firing=False
        #print(self.movestatus)
        if self.movestatus[1]==1 and self.fuel>0:
            self.docked=0
            self.F_rocket=(self.p_rocket*math.cos((self.angle+90)*math.pi/180),-self.p_rocket*math.sin((self.angle+90)*math.pi/180))
            self.fuel-=0.275
            self.firing=True
        if self.movestatus[2]==1 and self.docked ==0:
            self.angular_velocity += -0.016
        if self.movestatus[0]==1 and self.docked ==0:
            self.angular_velocity += 0.016
        #print(self.movestatus)
        #print(self.movestatus) 
        self.movestatus=[0,0,0,0]    
         
        
        if self.docked==0:   

            self.rot()    
            #Update speeds
            self.velocity=(self.velocity[0]+self.F_gravity[0]+self.F_rocket[0],
                           self.velocity[1]+self.F_gravity[1]+self.F_rocket[1])
            self.v2=self.velocity[0]**2+self.velocity[1]**2
            
            
            C_d=0.05
            C_d_a=0.005
            #Apply drag
            self.velocity=(self.velocity[0]*(1-1*C_d*rho*(self.velocity[0]**2)/2),self.velocity[1]*(1-1*C_d*rho*(self.velocity[1]**2)/2))
            self.angular_velocity=self.angular_velocity*(1-1*C_d_a*rho*(self.angular_velocity**2)/2)
            
            #Angles to feed to the DDQN input layer:
            #All angles are given in the cylindrical coordinate system centered on the planets CoG.
            F_gravity_norm=self.norm(self.F_gravity)
            velocity_norm=self.norm(self.velocity)
            grade_vector_norm=self.norm(self.grade_vector)
            
            #Velocity vector
            v1_u = self.velocity / velocity_norm
            v2_u = self.F_gravity / F_gravity_norm
            self.theta_vel=np.arccos(self.clamp(self.dot(v1_u, v2_u),-1,1))
           
            #spaceship in ref to prograde/retrograde
            self.grade_vector=(-np.sin((self.angle)*math.pi/180),-np.cos(self.angle*math.pi/180))
            v1_u =  self.grade_vector/ grade_vector_norm
            v2_u = self.velocity / velocity_norm
            self.theta_prograde=np.arccos(self.clamp(self.dot(v1_u, v2_u),-1,1))
            #print(self.theta_prograde)
                
            #Spaceship in ref to gravitational force
            v1_u = self.grade_vector / grade_vector_norm
            v2_u = self.F_gravity / F_gravity_norm
            self.theta_grav=np.arccos(self.clamp(self.dot(v1_u, v2_u),-1,1))
            #print( self.theta_grav)
           
            #theta_traveled
            gravity_vector=((self.position[0]-CoG[0]),(self.position[1]-CoG[1]))
            v1_u = gravity_vector / np.linalg.norm(gravity_vector)
            v2_u = (0,-1)
            self.theta_traveled=np.arccos(self.clamp(self.dot(v1_u, v2_u),-1,1))
            
             
                
                
            self.fps_counter+=1
            if self.fps_counter>2: #Rocket path trace density
                self.fps_counter=0
                self.old_positions.append(self.position)
        self.position=(self.position[0]+self.velocity[0],self.position[1]+self.velocity[1])
        self.rect.center=(self.position)
    def crash(self):
        self.velocity =(0,0)
        self.firing=False
        self.crashed=True
       
    def draw_flame(self,surface):
        if self.firing==True:
            surface.blit(self.flame, self.rect)


      
    def draw(self, surface,CoG):
        surface.blit(self.image, self.rect)
        #Draw the rocket path trace
        for i in self.old_positions:
                if i[0]>0: pygame.draw.circle(surface, (0,0,0), i, 1, 1)

