from dataclasses import dataclass
import numpy as np
import math
import matplotlib.pyplot as plt

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
    time: list[float]

class ControlLaw:
    def __init__(self, innerspeed, outerspeed, speed):
        self.lastSwitchTheta = None
        self.curr_inner = True
        self.inner_speed = innerspeed
        self.outer_speed = outerspeed
        self.speed = speed
        
    def straight_line(self, state):
        return Control(self.speed, self.speed)
        
    def circle(self, state):
        if self.curr_inner == False:
            return Control(self.inner_speed, self.outer_speed)
        return Control(self.outer_speed, self.inner_speed)

    def figure_8(self, state):

        if self.lastSwitchTheta == None:
            self.lastSwitchTheta = state.theta
        if abs(self.lastSwitchTheta-state.theta) < 2*math.pi:
            return self.circle(state)
        else:
            self.lastSwitchTheta = state.theta
            self.curr_inner = not self.curr_inner
            return self.circle(state)
            
        

class Simulator:
    def __init__(self, init_state, params, innerspeed, outerspeed, speed, dt):
        self.dt = dt
        self.controller = ControlLaw(innerspeed, outerspeed, speed)
        self.state = init_state
        self.parameters = params
        self.trajectory = Trajectory(
            parameters=params,
            controls = [],
            states = [init_state],
            time = []
        )
        
    
    def dynamics(self, ctrl, curr_state):
        return State(
            math.cos(curr_state.theta) * ((self.parameters.r * ctrl.rho1)/2 + (self.parameters.r * ctrl.rho2)/2),
            math.sin(curr_state.theta) * ((self.parameters.r * ctrl.rho1)/2 + (self.parameters.r * ctrl.rho2)/2),
            (self.parameters.r * ctrl.rho1)/(2*self.parameters.l) - (self.parameters.r * ctrl.rho2)/(2*self.parameters.l)
        )

    # use euler
    def integrate(self, curr_state, derivative):
        return State(
            curr_state.x + derivative.x * self.dt, 
            curr_state.y + derivative.y * self.dt, 
            curr_state.theta + derivative.theta * self.dt
        )

    # def create_trajectory(self):
    #     pass

    def plot(self):
        x_vals = [state.x for state in self.trajectory.states]
        y_vals = [state.y for state in self.trajectory.states]
        plt.figure(figsize=(8,4))
        plt.plot(x_vals, y_vals)
        plt.show()

    def run_all(self, ctrl, time_in_secs, resolution_per_sec):
        timer = np.linspace(0, time_in_secs, int(time_in_secs * resolution_per_sec))

        # current simulation i want to run
        
        for time in timer:
            self.trajectory.time.append(time)

            current_ctrl = ctrl(self.state)
            self.trajectory.controls.append(current_ctrl)
            
            derivative = self.dynamics(current_ctrl, self.state)
            self.state = self.integrate(self.state, derivative)
            self.trajectory.states.append(self.state)
        
        return self.trajectory
            



        
my_sim = Simulator(State(0,0,0), Parameters(2,4), 2, 5, 5, .05)
my_sim.run_all(my_sim.controller.figure_8, 10, 100)
my_sim.plot()