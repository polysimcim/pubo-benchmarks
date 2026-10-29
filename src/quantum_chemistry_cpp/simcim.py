import numpy as np
import torch
import torch.sparse

import argparse
import sys
import os

import time

DEVICE = torch.device('cpu')
DATATYPE = torch.float32


class SimCIM:
    """A class for k-local SimCIM implementation"""
    def __init__(self,
                 t_spins_num=0,
                 t_locality=0,
                 t_tensors=None,
                 t_iters_num=0,
                 t_time_step=0.0,
                 t_attempts_num=0,
                 t_saturation_value=0.0,
                 t_zeta=0.0,
                 t_pump_x_stretch=0.0,
                 t_pump_y_stretch=0.0,
                 t_pump_y_displ=0.0,
                 t_sigma=0.0,
                 t_ma_weight=0.0,
                 t_device=torch.device('cpu'),
                 t_datatype=torch.float32):
        """

        Args:
            t_spins_num (int):
            t_locality (int):
            t_tensors (list):
            t_iters_num (int):
            t_time_step (float):
            t_attempts_num (int):
            t_saturation_value (float):
            t_zeta (float):
            t_pump_x_stretch (float):
            t_pump_y_stretch (float):
            t_pump_y_displ (float):
            t_sigma (float):
            t_ma_weight (float):
            t_device (torch.device):
            t_datatype (torch.datatype)

        """
        self._spins_num = t_spins_num
        self._locality = t_locality

        self._tensors = t_tensors
        self._tensors[1] = self._tensors[1].view(self._spins_num, 1)

        self._iters_num = t_iters_num
        self._time_step = t_time_step
        self._attempts_num = t_attempts_num

        self._saturation_value = t_saturation_value
        self._zeta = t_zeta
        self._pump_x_stretch = t_pump_x_stretch
        self._pump_y_stretch = t_pump_y_stretch
        self._pump_y_displ = t_pump_y_displ

        self._sigma = t_sigma
        self._ma_weight = t_ma_weight

        self._device = t_device
        self._datatype = t_datatype

        self._einsum_attemptwise_strings = {k: self.generate_einsum_k_str_attemptwise(k) for k in range(1, self._locality + 1)}

        #self._tensors_norm = max([torch.max(torch.abs(self._tensors[k])) for k in range(self._locality + 1)])
        self._tensors_norm = torch.max(torch.abs(self._tensors[1]))
        for k in range(1, self._locality + 1):
            self._tensors[k] *= float(k) / self._tensors_norm.item()

        if self._locality >= 2:
            self._tensors_scale = max([torch.max(torch.sum(torch.abs(self._tensors[k]), list(range(1, k)))) for k in range(2, self._locality + 1)])
        else:
            self._tensors_scale = 1.0

    def anneal(self):
        amplitudes, _ = self.perform_amplitudes_evolution(store_evolution=True)
        spins = torch.sign(amplitudes)
        energy = torch.zeros(self._attempts_num, dtype=self._datatype).to(self._device)
        energy += self._tensors[0]
        for k in range(1, self._locality + 1):
            energy -= torch.einsum(self.generate_einsum_k_str_attemptwise_to_number(k), (self._tensors_norm.item() / float(k)) * self._tensors[k], *([spins] * k))

        return spins[:, torch.argmin(energy)], torch.min(energy)

    def perform_amplitudes_evolution(self, store_evolution=False):
        x = torch.zeros(self._spins_num, self._attempts_num, dtype=self._datatype).to(self._device)
        #x = 0.5 + 0.1 * torch.randn(self._spins_num, self._attempts_num, dtype=self._datatype).to(self._device)
        pump_values = self.calculate_pump_values()
        np.savetxt('./k_local_pump_values.txt', pump_values.numpy())
        delta_x_wma = torch.zeros((self._spins_num, self._attempts_num), dtype=self._datatype).to(self._device)

        if store_evolution:
            random_attempt = np.random.randint(self._attempts_num)
            evolution = torch.zeros([self._spins_num, self._iters_num], dtype=self._datatype).to(self._device)

        for t in range(self._iters_num):
            # print(f't = {t}')
            delta_x = self.calculate_delta_x(x, pump_values[t])
            # print('delta x: ')
            # print(delta_x)

            delta_x_wma = self._ma_weight * delta_x_wma + (1 - self._ma_weight) * delta_x
            # print('delta x wma: ')
            # print(delta_x_wma)

            x += delta_x_wma
            # print('x after adding delta, but before saturation: ')
            # print(x)

            x = self.saturate_amplitudes(x)
            # print('x after saturation')
            # print(x)

            if store_evolution:
                evolution[:, t] = x[:, random_attempt]

        if store_evolution:
            np.savetxt('./k_local_evol.txt', evolution.numpy())
            return x, evolution
        else:
            return x, None

    def calculate_delta_x(self, t_x, t_pump_value):
        return t_pump_value * t_x \
        + self._zeta * self.calculate_sum_of_einsums(t_x) \
        + self._sigma * torch.randn((self._spins_num, self._attempts_num), dtype=self._datatype).to(self._device)

    def calculate_sum_of_einsums(self, t_x):
        result = torch.zeros(self._spins_num, self._attempts_num, dtype=self._datatype)
        for k in range(1, self._locality + 1):
            result += torch.einsum(self._einsum_attemptwise_strings[k], self._tensors[k], *([t_x] * (k - 1)))

        return result

    def calculate_pump_values(self):
        time_points = torch.arange(self._iters_num, dtype=self._datatype).to(self._device)
        pump_arguments = torch.tensor(self._pump_x_stretch, dtype=self._datatype).to(self._device) * ((time_points / self._iters_num) - 0.5)

        return self._tensors_scale * self._pump_y_stretch * (torch.tanh(pump_arguments) + self._pump_y_displ)

    def saturate_amplitudes(self, t_x):
        saturation_test = (torch.abs(t_x) < self._saturation_value).type(self._datatype).to(self._device)

        return t_x * saturation_test + (1 - saturation_test) * self._saturation_value * torch.sign(t_x)

    @staticmethod
    def generate_einsum_k_str(t_k):
        """

        Args:
            t_k ():

        Returns:

        """
        indice_chars = [chr(ord('i') + idx) for idx in range(t_k)]
        einsum_k_str = ''.join(indice_chars)
        if t_k > 1:
            einsum_k_str += ',' + ','.join(indice_chars[1:])
        einsum_k_str += '->i'

        return einsum_k_str

    @staticmethod
    def generate_einsum_k_str_attemptwise(t_k):
        """

        Args:
            t_k ():

        Returns:

        """
        indice_chars = [chr(ord('i') + idx) for idx in range(t_k)]
        einsum_k_str_attemptwise = '...'
        einsum_k_str_attemptwise += ''.join(indice_chars)
        if t_k > 1:
            einsum_k_str_attemptwise += ',' + ','.join([char + 'h' for char in indice_chars[1:]])
        einsum_k_str_attemptwise += '->...i'
        if t_k > 1:
            einsum_k_str_attemptwise += 'h'

        return einsum_k_str_attemptwise

    @staticmethod
    def generate_einsum_k_str_attemptwise_to_number(t_k):
        """

        Args:
            t_k ():

        Returns:

        """
        indice_chars = [chr(ord('i') + idx) for idx in range(t_k)]
        einsum_k_str_attemptwise = ''.join(indice_chars)
        einsum_k_str_attemptwise += '...'
        einsum_k_str_attemptwise += ',' + ','.join([char + 'h' for char in indice_chars])
        einsum_k_str_attemptwise += '->h'

        return einsum_k_str_attemptwise


def dense_two_local_simcim():
    # Create the parser
    parser = argparse.ArgumentParser(description='Parser for one required SimCIM argument')

    # Add the arguments
    parser.add_argument('--path_to_working_dir',
                        nargs='?',
                        type=str,
                        help='Path to a directory where SimCIM should be applied',
                        default='./')

    # Execute the parse_args() method
    args = parser.parse_args()
    path_to_ising_constant = os.path.join(args.path_to_working_dir, '0.txt')
    path_to_ising_vector = os.path.join(args.path_to_working_dir, '1.txt')
    path_to_ising_matrix = os.path.join(args.path_to_working_dir, '2.txt')

    assert os.path.exists(path_to_ising_constant),\
        f'File for the Ising constant {path_to_ising_constant} does not exist'
    assert os.path.exists(path_to_ising_vector),\
        f'File for the Ising vector {path_to_ising_vector} does not exist'
    assert os.path.exists(path_to_ising_matrix),\
        f'File for the Ising matrix {path_to_ising_matrix} does not exist'

    ising_constant = np.loadtxt(path_to_ising_constant)
    ising_vector = np.loadtxt(path_to_ising_vector)
    ising_matrix = np.loadtxt(path_to_ising_matrix)

    spins_num = len(ising_vector)
    locality = 2

    tensors = [torch.tensor(ising_constant, dtype=DATATYPE).to(DEVICE),
               torch.tensor(ising_vector, dtype=DATATYPE).to(DEVICE),
               torch.tensor(ising_matrix, dtype=DATATYPE).to(DEVICE)]

    iters_num = 2000
    time_step = 0.5
    attempts_num = 20000

    saturation_value = 1.0
    zeta = 0.1
    pump_x_stretch = 3.0
    pump_y_stretch = 0.1
    pump_y_displ = 0.0

    sigma = 0.12
    ma_weight = 0.8

    device = DEVICE
    datatype = DATATYPE

    simcim = SimCIM(t_spins_num=spins_num,
                    t_locality=locality,
                    t_tensors=tensors,
                    t_iters_num=iters_num,
                    t_time_step=time_step,
                    t_attempts_num=attempts_num,
                    t_saturation_value=saturation_value,
                    t_zeta=zeta,
                    t_pump_x_stretch=pump_x_stretch,
                    t_pump_y_stretch=pump_y_stretch,
                    t_pump_y_displ=pump_y_displ,
                    t_sigma=sigma,
                    t_ma_weight=ma_weight,
                    t_device=device,
                    t_datatype=datatype)

    print(f'Annealing by k-local SimCIM starts. There are {ising_vector.shape[0]} spins in the system')
    start = time.time()
    ground_state, ground_energy = simcim.anneal()
    end = time.time()
    print(f'Annealing by SimCIM ended. Time elapsed: {end - start}')

    path_to_ground_state = os.path.join(args.path_to_working_dir, 'ground_state.txt')
    path_to_ground_energy = os.path.join(args.path_to_working_dir, 'ground_energy.txt')
    with open(path_to_ground_state, 'w') as ground_state_file:
        for elem in ground_state:
            ground_state_file.write(f'{0 if elem == 1 else 1}\n')
    with open(path_to_ground_energy, 'w') as ground_energy_file:
        ground_energy_file.write(f'{ground_energy.item()}')
    sys.exit(0)


def sparse_k_local_simcim():
    # Create the parser
    parser = argparse.ArgumentParser(description='Parser for one required SimCIM argument')

    # Add the arguments
    parser.add_argument('--spins_num',
                        type=int,
                        help='Locality the considered poly-Ising hamiltonian',
                        default=0)
    parser.add_argument('--locality',
                        type=int,
                        help='Locality the considered poly-Ising hamiltonian',
                        default=0)
    parser.add_argument('--path_to_working_dir',
                        type=str,
                        help='Path to a directory where SimCIM should be applied',
                        default='./')

    # Execute the parse_args() method
    args = parser.parse_args()


    spins_num = args.spins_num
    locality = args.locality

    tensors = []

    path_to_ising_constant = os.path.join(args.path_to_working_dir, '0.txt')
    ising_constant = -torch.tensor(np.loadtxt(path_to_ising_constant), dtype=DATATYPE).to(DEVICE)
    tensors.append(ising_constant)

    for k in range(1, locality + 1):
        path_to_tensor = os.path.join(args.path_to_working_dir, f'{k}.txt')
        assert os.path.exists(path_to_tensor), \
            f'File for the tensor {path_to_tensor} does not exist'
        np_tensor = np.loadtxt(path_to_tensor, ndmin=2)
        sparse_indices = torch.tensor(np_tensor[:, :k].astype(int).transpose())
        sparse_values = torch.tensor(np_tensor[:, k])
        tensor = torch.sparse.FloatTensor(sparse_indices,
                                          sparse_values,
                                          torch.Size([args.spins_num] * k)).to_dense().to(DEVICE)
        tensor = tensor.type(DATATYPE)
        tensors.append(tensor)

    iters_num = 2000
    time_step = 0.5
    attempts_num = 2000

    saturation_value = 1.0
    zeta = 0.1
    pump_x_stretch = 3.0
    pump_y_stretch = 0.1
    pump_y_displ = 0.0

    sigma = 0.12
    ma_weight = 0.8

    device = DEVICE
    datatype = DATATYPE

    simcim = SimCIM(t_spins_num=spins_num,
                    t_locality=locality,
                    t_tensors=tensors,
                    t_iters_num=iters_num,
                    t_time_step=time_step,
                    t_attempts_num=attempts_num,
                    t_saturation_value=saturation_value,
                    t_zeta=zeta,
                    t_pump_x_stretch=pump_x_stretch,
                    t_pump_y_stretch=pump_y_stretch,
                    t_pump_y_displ=pump_y_displ,
                    t_sigma=sigma,
                    t_ma_weight=ma_weight,
                    t_device=device,
                    t_datatype=datatype)

    print(f'Annealing by k-local SimCIM starts. There are {spins_num} spins in the system')
    start = time.time()
    ground_state, ground_energy = simcim.anneal()
    end = time.time()
    print(f'Annealing by SimCIM ended. Time elapsed: {end - start}')

    path_to_ground_state = os.path.join(args.path_to_working_dir, 'ground_state.txt')
    path_to_ground_energy = os.path.join(args.path_to_working_dir, 'ground_energy.txt')
    with open(path_to_ground_state, 'w') as ground_state_file:
        for elem in ground_state:
            ground_state_file.write(f'{0 if elem == 1 else 1}\n')
    with open(path_to_ground_energy, 'w') as ground_energy_file:
        ground_energy_file.write(f'{ground_energy.item()}')
    sys.exit(0)


def main():
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

    spins_num = len(ising_vector)
    locality = 2

    tensors = [torch.tensor(ising_constant, dtype=DATATYPE).to(DEVICE),
               torch.tensor(ising_vector, dtype=DATATYPE).to(DEVICE),
               torch.tensor(ising_matrix, dtype=DATATYPE).to(DEVICE)]

    iters_num = 2000
    time_step = 0.5
    attempts_num = 20000

    saturation_value = 1.0
    zeta = 0.1
    pump_x_stretch = 3.0
    pump_y_stretch = 0.2
    pump_y_displ = 0.0

    sigma = 0.12
    ma_weight = 0.8

    device = DEVICE
    datatype = DATATYPE

    simcim = SimCIM(t_spins_num=spins_num,
                    t_locality=locality,
                    t_tensors=tensors,
                    t_iters_num=iters_num,
                    t_time_step=time_step,
                    t_attempts_num=attempts_num,
                    t_saturation_value=saturation_value,
                    t_zeta=zeta,
                    t_pump_x_stretch=pump_x_stretch,
                    t_pump_y_stretch=pump_y_stretch,
                    t_pump_y_displ=pump_y_displ,
                    t_sigma=sigma,
                    t_ma_weight=ma_weight,
                    t_device=device,
                    t_datatype=datatype)
    print(f'Annealing by k-local SimCIM starts. There are {ising_vector.shape[0]} spins in the system')
    start = time.time()
    ground_state, ground_energy = simcim.anneal()
    end = time.time()
    print(f'Annealing by SimCIM ended. Time elapsed: {end - start}')
    print(f'Ground energy: {ground_energy}')
    print('Ground state:')
    print(ground_state)


if __name__ == '__main__':
    sparse_k_local_simcim()
