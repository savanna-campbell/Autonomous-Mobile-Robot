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
    beta: float # beta term in tanh

@dataclass
class Control:
    u_R: float # normalized right PWM command
    u_L: float # normalized left PWM command
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
    def __init__(self, fwd_speed, turn_rad, params):
        self.lastSwitchTheta = None
        self.fwd_speed = fwd_speed
        self.turn_radius = turn_rad
        self.parameters = params
    
    # takes target speeds and decides PWM speeds
    def find_speeds(self, right_speed, left_speed):
        damping_R = self.parameters.ka * right_speed
        damping_L = self.parameters.ka * left_speed

        friction_R = self.parameters.kf * math.tanh(self.parameters.alpha * right_speed)
        friction_L = self.parameters.kf * math.tanh(self.parameters.alpha * left_speed)

        drag_R = self.parameters.kq * math.tanh(self.parameters.beta * right_speed) * (right_speed) * right_speed
        drag_L = self.parameters.kq * math.tanh(self.parameters.beta * left_speed) * left_speed * left_speed
        
        u_R = (damping_R + friction_R + drag_R) / self.parameters.kg_R
        u_L = (damping_L + friction_L + drag_L) / self.parameters.kg_L
        
        return Control(u_R, u_L)


    def straight_line(self, state):
        angular_speed_R = self.fwd_speed / self.parameters.r_R
        angular_speed_L = self.fwd_speed / self.parameters.r_L
        return self.find_speeds(angular_speed_R, angular_speed_L)
        
    def circle(self, state):
        right_speed = self.fwd_speed / self.parameters.r_R * (1 + self.parameters.l / self.turn_radius)
        left_speed = self.fwd_speed / self.parameters.r_L * (1 - self.parameters.l / self.turn_radius)
        
        return self.find_speeds(right_speed, left_speed)

    def turn_in_place(self, state):
        right_speed = (self.fwd_speed * self.parameters.l) / (self.parameters.r_R)
        left_speed = -((self.fwd_speed * self.parameters.l) / (self.parameters.r_L))
        
        return self.find_speeds(right_speed, left_speed)

    def figure_8(self, state):

        if self.lastSwitchTheta == None:
            self.lastSwitchTheta = state.theta

        if abs(self.lastSwitchTheta-state.theta) < 2*math.pi:
            return self.circle(state)
        
        else:
            self.lastSwitchTheta = state.theta
            self.turn_radius = -self.turn_radius
            return self.circle(state)
            
        

class Simulator:
    def __init__(self, init_state, params, fwd_speed, turning_radius, dt):
        self.dt = dt
        self.controller = ControlLaw(fwd_speed, turning_radius, params)
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

        drag_R = self.parameters.kq * math.tanh(self.parameters.alpha * curr_state.omega_R) * curr_state.omega_R * curr_state.omega_R
        drag_L = self.parameters.kq * math.tanh(self.parameters.alpha * curr_state.omega_L) * curr_state.omega_L * curr_state.omega_L

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

    

    
# NEEDS GRAPH TITLES
    def plot(self):
        x_vals = [state.x for state in self.trajectory.states]
        y_vals = [state.y for state in self.trajectory.states]
        plt.figure(figsize=(6,6))
        plt.plot(x_vals, y_vals)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Robot Trajectory")
        plt.show()

# IN PROGRESS
    def run_all(self, ctrl, time_in_secs):
        timer = np.arange(0, time_in_secs, self.dt)

        # current simulation i want to run
        
        for time in timer:
            # add the time to trajectory
            self.trajectory.time.append(time)

            # append state
            self.trajectory.states.append(self.state)


            # add the controls to trajectory
            current_ctrl = ctrl(self.state)
            self.trajectory.controls.append(current_ctrl)

            # grab wheel speeds from dynamics
            w_R, w_L = self.dynamics(current_ctrl, self.state)

            # run kinematics to update state
            self.state = self.kinematics(self.state, w_R, w_L)
            
        self.export_CSV(self.trajectory)
        
        return self.trajectory

# NEEDS PARAMETERS, STATES, AND CONTROLS UPDATED
    def export_CSV (self, traj):
        with open("output.csv", "w", newline="") as f:
            f.write(f"# r_R = {traj.parameters.r_R}\n")
            f.write(f"# r_L = {traj.parameters.r_L}\n")
            f.write(f"# l = {traj.parameters.l}\n")
            f.write(f"# kg_R = {traj.parameters.kg_R}\n")
            f.write(f"# kg_L = {traj.parameters.kg_L}\n")
            f.write(f"# ka = {traj.parameters.ka}\n")
            f.write(f"# kf = {traj.parameters.kf}\n")
            f.write(f"# kq = {traj.parameters.kq}\n")
            f.write(f"# alpha = {traj.parameters.alpha}\n")
            f.write(f"# beta = {traj.parameters.beta}\n")

            writer = csv.DictWriter(f, fieldnames=["time", "x", "y", "theta", "omega_R", "omega_L", "u_R", "u_L"])
            writer.writeheader()

            for t, state, control in zip(traj.time, traj.states, traj.controls):
                writer.writerow({
                    "time": t,
                    "x": state.x,
                    "y": state.y,
                    "theta": state.theta,
                    "omega_R": state.omega_R,
                    "omega_L": state.omega_L,
                    "u_R": control.u_R,
                    "u_L": control.u_L,
                })



# Test case
test_params = Parameters(
    r_R=0.03, r_L=0.03, l=0.08,
    kg_R=25.0, kg_L=25.0,
    ka=0.5, kf=0.15, kq=0.02, 
    alpha=8.0, beta=8.0
)

fwd_speed = 0.5   # m/s
turn_radius = 0.3  # m
dt = 0.005
time = 100

my_sim = Simulator(State(0,0,0,0,0), test_params, fwd_speed, turn_radius, dt)
my_sim.run_all(my_sim.controller.turn_in_place, time)
my_sim.plot()