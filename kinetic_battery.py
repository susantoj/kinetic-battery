#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Kinetic Battery Model Library
Copyright (C) 2014 Julius Susanto

Last edited: November 2014

Main Functions
--------------
- capacity_step: calculates the battery capacity at the next time step
- estimate_constants: estimates the battery constants k, c and qmax based on
                      battery charge or discharge data

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np

import os
import matplotlib.pyplot as plt

def capacity_step(q1_0, q2_0, k, c, qmax, i, dt):
    """
    Returns the available and bound charges for the next time step
    Based on the Manwell and McGowan Kinetic Battery Model
    
    Inputs: 
        q1_0    Available charge at beginning of time step (Ah)
        q2_0    Bound charge at beginning of time step (Ah)
        k       Battery rate constant (rate at which chemically bound charge becomes available)
        c       Battery capacity ratio (fraction of total charge that is available)
        qmax    Maximum amount of charge in battery (Ah)
        i       Charge (-) or discharge (+) current of battery (A)
        dt      Length of time step (hours)
    
    Outputs:
        q1      Available charge at next time step (Ah)
        q2      Bound charge at next time step (Ah)
    """
    r = np.exp(-k * dt)
    q0 = q1_0 + q2_0
    
    # Calculate maximum discharge and charging currents
    id_max = (k * q1_0 * r + q0 * k * c * (1 - r)) / (1 - r + c * (k * dt - 1 + r))
    ic_max = (-k * c * qmax + k * q1_0 * r + q0 * k * c * (1 - r)) / (1 - r + c * (k * dt - 1 + r))
    
    # Check if battery current is within maximum bounds
    # If not, set battery current to limits
    if i > id_max:
        i = id_max
    if i < ic_max:
        i = ic_max
    
    q1 = q1_0 * r + ((q0 * k * c - i) * (1 - r) - i * c * (k * dt - 1 + r)) / k
    q2 = q2_0 * r + q0 * (1 - c) * (1 - r) - i * (1 - c) * (k * dt - 1 + r) / k
    
    return q1, q2

def estimate_constants(x0, I, T, err_tol, max_iter):
    """
    Estimates the battery constants k, c and qmax using a non-linear least squares algorithm
    
    Inputs: 
        x0          Initial guess vector [k0, c0, qmax0]
        I           Vector of discharge (or charge) currents (A)
        T           Vector of discharge (or charge) times associated with I (hours)
        err_tol     Error tolerance for least squares algorithm
        max_iter    Maximum number of iterations
    
    Outputs:
        x       Vector of estimated constants [k, c, qmax]
        conv    Converged (True/False)
        iter    Number of iterations
        err     Least squares error
    """
    
    n = len(I)
    if n != len(T) or n < 3:
        print('Vectors I and T are not of same size or are of length < 3')
        return x0, False, 0, 9999
        
    x = x0
    q_sol = np.multiply(I,T)    # Solution vector
    
    # Construct initial mismatch vector dq
    k = x[0]
    c = x[1]
    qmax = x[2]
    exp_T = np.exp(-k * np.array(T))
    q_num = qmax * k * c * np.array(T)
    q_den = np.ones(n) - exp_T + c * (k * np.array(T) - np.ones(n) + exp_T)
    q = np.divide(q_num, q_den)
    dq = q_sol - q
    err = np.sqrt(np.dot(dq, dq.T))
    
    iter = 1
    while (err > err_tol) and (iter < max_iter):
        # Construct Jacobian matrix
        dq_dk_num = np.multiply(q_den, qmax * c * np.array(T)) - np.multiply(qmax * k * c * np.array(T), c * np.array(T) + k * exp_T * (1 - c))
        dq_den = np.multiply(q_den, q_den)
        dq_dk = np.divide(-dq_dk_num, dq_den)
        
        dq_dc_num = np.multiply(q_den, qmax * k * np.array(T)) - np.multiply(qmax * k * c * np.array(T), k * np.array(T) - np.ones(n) + exp_T) 
        dq_dc = np.divide(-dq_dc_num, dq_den)
        
        dq_dqmax = -q / qmax
        
        J = np.vstack([dq_dk, dq_dc, dq_dqmax]).T
        
        # Solve non-linear least squares iteration
        # dx = (J'J)^(-1) J' dq
        jmat = np.dot(np.matrix(np.dot(J.T, J)).I, J.T)
        dx = np.dot(jmat, dq.T).A[0]        
        x = x - dx
        
        # Construct mismatch vector
        k = x[0]
        c = x[1]
        qmax = x[2]
        exp_T = np.exp(-k * np.array(T))
        q_num = qmax * k * c * np.array(T)
        q_den = np.ones(n) - exp_T + c * (k * np.array(T) - np.ones(n) + exp_T)
        q = np.divide(q_num, q_den)
        dq = q_sol - q
        err = np.sqrt(np.dot(dq, dq.T))
        
        iter = iter + 1
    
    # Check convergence
    if err < err_tol:
        conv = True
    else:
        conv = False
    
    return x, conv, iter, err