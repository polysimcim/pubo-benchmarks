import numpy as np
import torch

import argparse
import sys
import os

import time

device = torch.device("cpu")
datatype = torch.float32


# amplitude increment
def ampl_inc(J, b, c, zeta, p, sigma, attempt_num, dt, device, datatype):
    return ((p*c + zeta*(torch.mm(J, c)+b))*dt
            + (sigma*torch.randn((c.size(0), attempt_num), dtype=datatype).to(device))*dt)
#     return ((p*c + zeta*(torch.mm(J,c)))*dt
#             + (sigma*torch.randn((c.size(0),attempt_num),dtype=datatype).to(device))*dt)


# pump ramp
def pump(O, S, D, Jmax, N, device, datatype):
    i = torch.arange(N, dtype=datatype).to(device)
    arg = torch.tensor(S, dtype=datatype).to(device)*(i/N-0.5)
    return Jmax * O * (torch.tanh(arg) + D)

# # coupling ramp
# def coupling(init_value,final_value,dt,N):
#     i = torch.arange(N).type(datatype).to(device)
#     k = (final_value - init_value)/N/dt
#     return init_value + k*i*dt


# amplitude initializer 
def init_ampl(dim, attempt_num, device, datatype):
    return 0.5 + 0.1 * torch.randn((
        dim, attempt_num), dtype=datatype).to(device)


def sigmoid(x):
    return 1. / (1. + torch.exp(-x))


def f(x,c_th,device,datatype):
    th_test = (torch.abs(x)<c_th).type(datatype).to(device)
    return x*th_test + (1-th_test)*torch.sign(x)*c_th


def g(x,c_th,device,datatype):
    th_test = (x<c_th).type(datatype).to(device)
    zero_test = (x>0.).type(datatype).to(device)
    return x*th_test*zero_test + (1-th_test)*c_th + 0.*(1-zero_test)


# evolution of amplitudes
# N -- number of time iterations
# attempt_num -- number of runs 
# J -- coupling matrix
# b -- biases
# O, S, D -- pump parameters
# sigma -- sqrt(std) for random number
# alpha -- momentum parameter
# c_th -- restriction on amplitudes growth
def evolve(N,attempt_num,dt,J,b,O,S,D,c_th,sigma,zeta,alpha,device,datatype):
    dim = J.size(0)
    #print(J)
    Jmax = torch.max(torch.sum(torch.abs(J),1))
    print(f'Tensors scale = {Jmax}')
    
    # choosing an attempt for the amplitude evolution 
    random_attempt = np.random.randint(attempt_num)
    
    # initializing current amplitudes
    c_current = init_ampl(dim, attempt_num,device,datatype)
    
    # creating the array for evolving amplitudes from random attempt
    c_evol = torch.empty((dim, N),dtype=datatype).to(device)
    c_evol[:,0] = c_current[:,random_attempt]
    
    # define pump array
    #print(Jmax)
    p = pump(O,S,D,Jmax,N,device,datatype)
    np.savetxt("./python_pump.txt", p.numpy())
    
    # define copupling growth
#     zeta_b = coupling(init_value,final_value,dt,N)
    
    # initializing moving average of amplitudes increment
    dc_momentum = torch.zeros((dim, attempt_num),dtype=datatype).to(device)
    for i in range(1,N):
        
        # calculating amplitude increment
        dc = ampl_inc(J,b,c_current,zeta,p[i],sigma,attempt_num,dt,device,datatype)
        
        # calculating moving average of amplitudes increment
        dc_momentum = alpha*dc_momentum + (1-alpha)*dc
        
#         # calculating possible values of amplitudes on the next step
        c1 = c_current + dc_momentum
        
#         # comparing c1 with c_th
#         th_test = (torch.abs(c1)<c_th).type(datatype)
        
        # updating c_current
#         c_current = sigmoid(c_current + dc_momentum)
#         c_current = th_test*(c_current + dc_momentum) + (1.-th_test)*torch.sign(c_current)*c_th
        c_current = f(c1,c_th,device,datatype)
#         c_current = sigmoid(c_current + dc_momentum)
        
        # add amplitude values from random attempt to c_evol array 
        c_evol[:,i] = c_current[:,random_attempt]

    np.savetxt("./python_evol.txt", c_evol.numpy())
    return c_current, c_evol


def annealer(G, h, C):
    dim = G.shape[0]
    Norm = np.max([np.max(np.abs(h)), np.max(np.abs(h))])
    print(f'Tensors norm = {Norm}')

    J = torch.tensor(2*G/Norm, dtype=datatype).to(device)
    b = torch.tensor(h/Norm, dtype=datatype).to(device).reshape(dim, 1)
    const = torch.tensor(C, dtype=datatype).to(device)

    c_th = 1.
    zeta = 0.1
    N = 2000
    attempt_num = 20000
    dt = 0.5
    sigma = 0.12
    alpha = 0.8
    Jmax = torch.max(torch.sum(torch.abs(J),1))

    S = 3.
    D = -0.
    O = 0.2

    c_current, c_evol = evolve(N,
                               attempt_num,
                               dt,
                               J,
                               b,
                               O,
                               S,
                               D,
                               c_th,
                               sigma,
                               zeta,
                               alpha,
                               device,
                               datatype)
    spins_current = torch.sign(c_current)
    spins_evol = torch.sign(c_evol)
    energy =  (const + (-0.5*torch.einsum('ij,ik,jk->k', (Norm*J, spins_current, spins_current))
          - torch.einsum('ij,ik->k', (Norm*b,spins_current))))
    return spins_current[:,torch.argmin(energy)]


def main():
    # Create the parser
    # parser = argparse.ArgumentParser(description='Parser for one required SimCIM argument')
    #
    # # Add the arguments
    # parser.add_argument('--path_to_working_dir',
    #                     nargs='?',
    #                     type=str,
    #                     help='Path to a directory where SimCIM should be applied',
    #                     default='./')
    #
    # # Execute the parse_args() method
    # args = parser.parse_args()
    # path_to_ising_constant = os.path.join(args.path_to_working_dir, '0.txt')
    # path_to_ising_vector = os.path.join(args.path_to_working_dir, '1.txt')
    # path_to_ising_matrix = os.path.join(args.path_to_working_dir, '2.txt')
    #
    # assert os.path.exists(path_to_ising_constant),\
    #     f'File for the Ising constant {path_to_ising_constant} does not exist'
    # assert os.path.exists(path_to_ising_vector),\
    #     f'File for the Ising vector {path_to_ising_vector} does not exist'
    # assert os.path.exists(path_to_ising_matrix),\
    #     f'File for the Ising matrix {path_to_ising_matrix} does not exist'
    #
    # ising_constant = np.loadtxt(path_to_ising_constant)
    # ising_vector = np.loadtxt(path_to_ising_vector)
    # ising_matrix = np.loadtxt(path_to_ising_matrix)

    path_to_working_dir = os.path.join('./../',
                                       'tensors_np_readable',
                                       'hydrogen',
                                       '1_95.txt',
                                       'r=1',
                                       '1')
    path_to_ising_constant = os.path.join(path_to_working_dir, '0.txt')
    path_to_ising_vector = os.path.join(path_to_working_dir, '1.txt')
    path_to_ising_matrix = os.path.join(path_to_working_dir, '2.txt')

    assert os.path.exists(path_to_ising_constant), \
        f'File for the Ising constant {path_to_ising_constant} does not exist'
    assert os.path.exists(path_to_ising_vector), \
        f'File for the Ising vector {path_to_ising_vector} does not exist'
    assert os.path.exists(path_to_ising_matrix), \
        f'File for the Ising matrix {path_to_ising_matrix} does not exist'

    ising_constant = np.loadtxt(path_to_ising_constant)
    ising_vector = np.loadtxt(path_to_ising_vector)
    ising_matrix = np.loadtxt(path_to_ising_matrix)

    print(f'Annealing by SimCIM starts. There are {ising_vector.shape[0]} spins in the system')
    start = time.time()
    ground_state = np.asarray(annealer(ising_matrix, ising_vector, ising_constant))
    end = time.time()
    print(f'Annealing by SimCIM ended. Time elapsed: {end - start}')
    ground_energy = ising_constant - np.dot(ising_vector, ground_state) - np.dot(np.dot(ground_state, ising_matrix), ground_state)
    print(f'Ground energy: {ground_energy}')
    print('Ground state:')
    print(ground_state)
    # path_to_ground_state = os.path.join(args.path_to_working_dir, 'ground_state.txt')
    # path_to_ground_energy = os.path.join(args.path_to_working_dir, 'ground_energy.txt')
    # with open(path_to_ground_state, 'w') as ground_state_file:
    #     for elem in ground_state:
    #         ground_state_file.write(f'{0 if elem == 1 else 1}\n')
    # with open(path_to_ground_energy, 'w') as ground_energy_file:
    #     ground_energy_file.write(f'{ground_energy}')
    # sys.exit(0)


if __name__ == '__main__':
    main()
