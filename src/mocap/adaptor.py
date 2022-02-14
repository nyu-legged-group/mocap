from mocap.qp import QpSolver
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

class MotionAdaptor(object):
  def __init__(self, nj, vel_bound, pos_bound_upper, pos_bound_lower, weightdict):
    '''
    A class to adapt data from Vicon motion data in human to humanoid
    by considering the position and velocity constraint and using second
    order dynamics
    '''
    
    self.nj = nj
    self.vel_bound = vel_bound
    self.pos_bound_upper = pos_bound_upper
    self.pos_bound_lower = pos_bound_lower
#     self.w_acceleration = 100000. #minimizing acceleration
#     self.w_tracking = 100. # track PD controller
#     self.w_vel = 100. # track velocity

#     self.p_tracking = 600000.
#     self.d_tracking = 200000.
    self.w_acceleration = weightdict["w_acceleration"] #minimizing acceleration
    self.w_tracking = weightdict["w_tracking"] # track PD controller
    self.w_vel = weightdict["w_vel"] # track velocity

    self.p_tracking = weightdict["p_tracking"]
    self.d_tracking = weightdict["d_tracking"]

    self.acc_des = np.zeros(self.nj)
    self.vel_des = np.zeros(self.nj)
    
    self.qp_solver = QpSolver()
    self.last_q = None
    self.last_dq = None
    self.dt = 1./240.
  
  def forward(self, q, dq, ddq):
    self.last_q = q.copy()
    self.last_dq = dq.copy()
    q += dq*self.dt + 0.5*ddq*self.dt**2
    dq += ddq*self.dt
    return q, dq

  def fill_weights(self):
    w1 = [self.w_acceleration * np.ones(self.nj)]
    w2 = [self.w_tracking * np.ones(self.nj)]
    w3 = [self.w_vel * np.ones(self.nj)]
    self.Q1 = np.diag(np.hstack(w1))
    self.Q2 = np.diag(np.hstack(w2))
    self.Q3 = np.diag(np.hstack(w3))

  def fill_vel_bound(self, dq):
    self.h_vel_upper = 1/self.dt*(self.vel_bound-dq)
    self.h_vel_lower = -1/self.dt*(-self.vel_bound-dq)

  def fill_pos_bound(self, q, dq):
    self.h_pos_upper = 2/self.dt**2*(self.pos_bound_upper-q-self.dt*dq)
    self.h_pos_lower = -2/self.dt**2*(self.pos_bound_lower-q-self.dt*dq)

  def fill_acc_des(self, q, dq, q_ref, dq_ref, ddq_ref):
    self.acc_des = ddq_ref + self.p_tracking*(q_ref-q)+self.d_tracking*(dq_ref-dq)
  
  def fill_vel_des(self, dq, dq_ref):
    self.vel_des = dq_ref - dq

  def compute(self, q, dq, q_ref, dq_ref, ddq_ref):
    '''
    Arguements:
      q: current state
      dq: current velocity
      q_ref: reference state
      dq_ref: reference velocity
      ddq_ref: reference acceleration
    
    Return:
      ddq: current acceleration

    Solve a Quadratic Program defined as:

            minimize
                (1/2) * x.T * P * x + q.T * x

            subject to
                G * x <= h 
                A * x == b
    Assume x is a n dimensional vector
    P matrix shape: nxn
    q matrix shape: 1xn
    G matrix shape: mxn
    h matrix shape: 1xm
    quadprog_solve_qp(self, P, q, G=None, h=None, A=None, b=None, initvals=None)

    '''

    self.fill_acc_des(q, dq, q_ref, dq_ref, ddq_ref)
    self.fill_vel_des(dq, dq_ref)
    self.fill_weights()
    self.fill_pos_bound(q, dq)
    self.fill_vel_bound(dq)

    P = self.Q1 + self.Q2 + self.dt**2*self.Q3
    q = -self.Q2.T.dot(self.acc_des)-self.dt*self.Q3.T.dot(self.vel_des)
    G = np.vstack([np.identity(self.nj), -1.*np.identity(self.nj),np.identity(self.nj), -1*np.identity(self.nj)])
    h = np.hstack([self.h_pos_upper, self.h_pos_lower, self.h_vel_upper, self.h_vel_lower])

    return self.qp_solver.quadprog_solve_qp(P, q, G, h)

