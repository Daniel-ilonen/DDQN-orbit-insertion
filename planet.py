#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 02:33:16 2023

@author: daniel
"""
import pygame 
class Planet(pygame.sprite.Sprite):
      def __init__(self,CoG):
        super().__init__() 
        self.radius=100
        self.image = pygame.image.load("img/earth.png")
        self.image =pygame.transform.scale(self.image, (self.radius*2,self.radius*2))
        self.rect = self.image.get_rect()
        self.CoG=CoG
        self.rect.center = self.CoG
        self.g=0.008
        self.ratio_atmos_earth=1.8
        
      def draw(self,surface, ratio_atmos_earth):
        surface.blit(self.image, self.rect) 
        pygame.draw.circle(surface, (0,0,255), (self.rect.center[0], self.rect.center[1]), self.radius*ratio_atmos_earth, 3)
        pygame.draw.circle(surface, (0,0,255), (self.rect.center[0], self.rect.center[1]), self.radius*ratio_atmos_earth, 3)
      def get_rho(self,height):
        #1 at radius, 0 at space limit, decrease lineraly
        l=self.radius 
        r=self.ratio_atmos_earth
        k=1/(l-r*l)
        m=1-l*k
        rho=height*k+m
        rho = rho if rho>0 else 0
        return rho
        #print(rho)