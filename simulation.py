from dataclasses import dataclass
import numpy as np
import math

# define parameters tracking radius and length
@dataclass
class Parameters:
    r: float
    l: float

@dataclass
class Control:
    rho1: float
    rho2: float

@dataclass
class State:
    x: float
    y: float
    theta: float

@dataclass
class Trajectory:
    parameters: Parameters
    controls: list[Control]
    states: list[State]
    time: np.ndarray

class ControlLaw:
    def __init__(self):
        self.lastSwitchTheta = None
        self.curr_inner = True
        

    def straight_line(self, speed):
        return Control(speed, speed)
        
    def circle(self, inner_wheel, inner_wheel_speed, outer_wheel_speed):
        if inner_wheel == False:
            return Control(inner_wheel_speed, outer_wheel_speed)
        return Control(outer_wheel_speed, inner_wheel_speed)

    def figure_8(self, currTheta, innerspeed, outerspeed):
        if self.lastSwitchTheta == None:
            self.lastSwitchTheta = currTheta
        if abs(self.lastSwitchTheta-currTheta) < math.pi:
            return self.circle(self.curr_inner, innerspeed, outerspeed)
        else:
            self.lastSwitchTheta = currTheta
            self.currInner = not self.currInner
            return self.circle(self.curr_inner, innerspeed, outerspeed)
            
        

class Simulator:
    def __init__(self, init_state, params):
        self.controller = ControlLaw()
        self.initial_state = init_state
        self.parameters = params
        
    
    def dynamics(self):
        pass

    def integrate(self):
        pass

    def create_trajectory(self):
        pass

    def plot(self):
        pass

    def run_all(self):
        pass