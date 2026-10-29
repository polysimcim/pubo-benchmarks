//
// Created by Aleksei Malyshev on 23/10/2019.
//

#ifndef SIMCIM_QUANTUM_CHEMISTRY_QUANTUM_CHEMISTRY_H
#define SIMCIM_QUANTUM_CHEMISTRY_QUANTUM_CHEMISTRY_H
#include <vector>
#include <utility>

#include "Hamiltonian.h"
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;
using fs::path;
using namespace std;

const double BIG_LAMBDA = 100;

void check_hydrogen_hamiltonian();

Hamiltonian derive_c_matrix(size_t t_qubit_num, size_t t_r_num, const vector<int>& t_signs);
Hamiltonian derive_c_matrix_fast(size_t t_qubit_num, size_t t_r_num, const vector<int>& t_signs);
void check_c_matrix_derivation(size_t t_qubit_num, size_t t_r_num, const vector<int>& t_signs);

void check_paper_example();
pair<double, string> solve_quantum_chemistry(const Hamiltonian& t_hamiltonian, size_t t_r_num, solution_method t_solution_method=BRUTE_FORCE);

#endif //SIMCIM_QUANTUM_CHEMISTRY_QUANTUM_CHEMISTRY_H
