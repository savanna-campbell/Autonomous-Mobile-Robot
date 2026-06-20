from dataclasses import dataclass
import dataclasses
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import numpy as np

# define parameters tracking radius and length
@dataclass
class Parameters:
    r_R: float # radius of right wheel
    r_L: float # radius of left wheel
    l: float # distance from wheel to center of mass
    kg_R: float # gain of right wheel
    kg_L: float # gain of left wheel
    ka: float # effective damping coefficient
    kf: float # effective friction coefficient
    kq: float  # effective drag coefficient
    alpha: float # alpha term in tanh

@dataclass
class Control:
    u_L: float # normalized left PWM command
    u_R: float # normalized right PWM command
    #rho1: float
    #rho2: float

@dataclass
class State:
    x: float
    y: float
    theta: float
    omega_R: float # angular velocity of right wheel
    omega_L: float # angular velocity of left wheel
   # v: float # velocity
   # omega: float # angular velocity (theta dot)

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
            states = [],
            time = []
        )

    # goal: update wheel velocity based on controls
    def dynamics(self, ctr, curr_state):
        gain_R = self.parameters.kg_R * ctr.u_R
        gain_L = self.parameters.kg_L * ctr.u_L

        damping_R = self.parameters.ka * curr_state.omega_R
        damping_L = self.parameters.ka * curr_state.omega_L

        friction_R = self.parameters.kf * math.tanh(self.parameters.alpha * curr_state.omega_R)
        friction_L = self.parameters.kf * math.tanh(self.parameters.alpha * curr_state.omega_L)
        drag_R = self.parameters.kq * abs(curr_state.omega_R) * curr_state.omega_R
        drag_L = self.parameters.kq * abs(curr_state.omega_L) * curr_state.omega_L
        omega_R_dot = gain_R - damping_R - friction_R - drag_R
        omega_L_dot = gain_L - damping_L - friction_L - drag_L
        
        omega_R = curr_state.omega_R + omega_R_dot * self.dt
        omega_L = curr_state.omega_L + omega_L_dot * self.dt
        
        return omega_R, omega_L

    # turns wheel velocities into their kinematic state
    def kinematics(self, curr_state, omega_R, omega_L):

        R_body_to_world = np.array([[math.cos(curr_state.theta), -math.sin(curr_state.theta), 0],
                          [math.sin(curr_state.theta), math.cos(curr_state.theta), 0],
                          [0, 0, 1]])
        
        local_frame = np.array([(self.parameters.r_R * omega_R / 2) + (self.parameters.r_L * omega_L / 2),
                                0,
                                (self.parameters.r_R * omega_R / (2 * self.parameters.l)) - (self.parameters.r_L * omega_L / (2 * self.parameters.l))])
        
        global_frame = R_body_to_world @ local_frame
        
        x_dot = global_frame[0]
        y_dot = global_frame[1]
        theta_dot = global_frame[2]

        return State(
            curr_state.x + x_dot * self.dt,
            curr_state.y + y_dot * self.dt,
            curr_state.theta + theta_dot * self.dt, 
            omega_R,
            omega_L 

        )

    

    

    def plot(self):
        x_vals = [state.x for state in self.trajectory.states]
        y_vals = [state.y for state in self.trajectory.states]
        plt.figure(figsize=(6,6))
        plt.plot(x_vals, y_vals)
        plt.show()

    def run_all(self, ctrl, time_in_secs):
        timer = np.linspace(0, time_in_secs, int(time_in_secs /self.dt))

        # current simulation i want to run
        
        for time in timer:
            self.trajectory.time.append(time)

            current_ctrl = ctrl(self.state)
            self.trajectory.controls.append(current_ctrl)
            
            derivative = self.dynamics(current_ctrl, self.state)
            self.state = self.integrate(self.state, derivative)
            self.trajectory.states.append(self.state)
        self.export_CSV(self.trajectory)
        
        return self.trajectory


    def export_CSV (self, traj):
        with open("output.csv", "w", newline="") as f:
            f.write(f"# r = {traj.parameters.r}\n")
            f.write(f"# l = {traj.parameters.l}\n")

            writer = csv.DictWriter(f, fieldnames=["time", "x", "y", "theta", "rho1", "rho2", "velocity", "omega"])
            writer.writeheader()

            for t, state, control in zip(traj.time, traj.states, traj.controls):
                writer.writerow({
                    "time": t,
                    "x": state.x,
                    "y": state.y,
                    "theta": state.theta,
                    "velocity": state.v,
                    "angular velocity": state.omega,
                    "rho1": control.rho1,
                    "rho2": control.rho2,
                })


my_sim = Simulator(State(0,0,0,0,0), Parameters(2,4), 2, 5, 5, .005)
my_sim.run_all(my_sim.controller.figure_8, 100)
my_sim.plot()