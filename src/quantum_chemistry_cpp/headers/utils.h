//
// Created by malyshev on 08/11/2019.
//

#ifndef SIMCIM_QUANTUM_CHEMISTRY_UTILS_H
#define SIMCIM_QUANTUM_CHEMISTRY_UTILS_H

#include <iostream>
#include <map>
#include <string>
#include <fstream>
#include <algorithm>

#include "./Hamiltonian.h"
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;
using fs::path;
using namespace std;

static map<solution_method, string> solution_method_to_str = {
        {BRUTE_FORCE,   "brute_force"},
        {EIGEN,         "eigen"},
        {SIMCIM_PYTHON, "simcim"},
        {SIMCIM_PYTHON_K_LOCAL, "simcim_python_k_local"},
        {SIMCIM_CPP, "simcim_cpp"},
        {SIMCIM_CPP_K_LOCAL, "simcim_cpp_k_local"},
};

static const double SANITY_QUBIT_NUM = 25;

Hamiltonian load_hamiltonian_from_pauli_strings(const path& t_path_to_hamiltonian);
Hamiltonian load_hydrogen_hamiltonian_from_pauli_strings(const path& t_path_to_hamiltonian);
void transform_hydrogen_hamiltonians_to_ising_matrices(const path& t_path_to_hamiltonians, const path& t_path_to_tensors);
void transform_hydrogen_hamiltonians_to_ising_tensors(const path& t_path_to_hamiltonians, const path& t_path_to_tensors);
void transform_hamiltonians_to_ising_matrices(const path& t_path_to_hamiltonians, const path& t_path_to_tensors);
void transform_hamiltonians_to_ising_tensors(const path& t_path_to_hamiltonians, const path& t_path_to_tensors);
void solve_hydrogen_hamiltonians(const path& t_path_to_hamiltonians, const path& t_path_to_solutions, const solution_method t_solution_method);
void solve_hamiltonians(const path& t_path_to_hamiltonians, const path& t_path_to_solutions, const solution_method t_solution_method);


#endif //SIMCIM_QUANTUM_CHEMISTRY_UTILS_H
