from vpython import *
from utils import *
import time as t
import numpy as  np

def simulate_cart_pole(r,config,controls):
    ### ADJUST CANVAS ###
    scene.background    = color.white
    scene.range         = 3
    scene.align         = 'left'
    ### CONSTANT ###
    print(r)
    cart_mass   = config["m1"]# kg
    pole_mass   = config["m2"]# kg
    pole_length = config["l"] # m
    lin_fric    = 0  
    rev_fric    = 0
    window      = 3

    ### OBJECT ###
    CART = CART_OBJECT(lin_fric,box(pos=vec(0, 0, 0), length=0.3, height=0.1, width=0.2, color=color.blue))
    CART.set_mass(cart_mass)
    POLE = POLE_OBJECT(rev_fric,box(length=pole_length, height=0.05, width=0.05,color=color.black),CART)
    POLE.set_mass(pole_mass)
    CART.attach_pole(POLE)
    POLE.update_pos()
    POLE.set_angle(-pi/2)

    ### SIMULATION ###
    for control in controls:
        rate(r)
        CART.set_force(control)
        CART.update_pos(1/r)
        POLE.update_pos()
        POLE.update_angle(1/r)