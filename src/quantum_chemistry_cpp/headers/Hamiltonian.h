//
// Created by Aleksei Malyshev on 18/10/2019.
//

#ifndef SIMCIM_QUANTUM_CHEMISTRY_HAMILTONIAN_H
#define SIMCIM_QUANTUM_CHEMISTRY_HAMILTONIAN_H

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <stack>
#include <utility>
#include <algorithm>

#include <stdio.h>

#include <chrono>

#include <boost/filesystem.hpp>

#include <Eigen/Eigenvalues>

#include "./Term.h"
#include "two_local_simcim.h"
#include "k_local_simcim.h"

//#define RECURSIVE_LOCALITY_REDUCTION 0

namespace fs = boost::filesystem;
using fs::path;
using namespace std;

using Eigen::VectorXcd;
using Eigen::VectorXd;
using Eigen::MatrixXd;

enum solution_method{EIGEN,
                     BRUTE_FORCE,
                     SIMCIM_PYTHON,
                     SIMCIM_PYTHON_K_LOCAL,
                     SIMCIM_CPP,
                     SIMCIM_CPP_K_LOCAL};

typedef pair<size_t, size_t> IndexPair;
typedef map<IndexPair, set<Term const*>> IndexPairTerms;

class Hamiltonian
{
    typedef void (Hamiltonian::*SolutionMethod)();

private:
    size_t m_qubit_num = 0;
    size_t m_dim = 0;
    Terms m_terms;

    std::unordered_map<solution_method, SolutionMethod> m_solution_methods;
    MatrixXcd m_eigenvectors;
    vector<complex<double>> m_eigenvalues;

    size_t m_ground_state_idx = 0xBAD;
    VectorXcd m_ground_state;
    double m_ground_state_energy = 0xBAD;

    bool m_is_solved = false;

public:
    // Constructors
    Hamiltonian() = default;
    explicit Hamiltonian(size_t t_qubit_num);
    explicit Hamiltonian(const Term& t_term);
    explicit Hamiltonian(const string& t_pauli_string, double t_weight = 1.0);
    void flush();

    // Getters
    size_t get_qubit_num() const  { return m_qubit_num; };
    const Terms& get_terms() const { return m_terms; };
    size_t get_terms_num() const  { return m_terms.size(); };
    size_t get_locality() const {  return (m_terms.empty()) ? 0:((*(m_terms.rbegin())).get_locality()); };
    MatrixXcd get_eigenvectors() const { return m_eigenvectors; };
    vector<complex<double>> get_eigenvalues() const { return m_eigenvalues; };
    double get_ground_state_idx() const { return m_ground_state_idx; };
    string get_ground_state_qubit_string() const;
    double get_diag_eigenvalue(size_t t_eigenvalue_idx) const;
    vector<double> get_diag_eigenvalues() const;
    double get_ising_constant() const {  return (m_terms.empty()) ? 0:((*(m_terms.begin())).get_weight()); };
    VectorXd get_ising_vector() const;
    MatrixXd get_ising_matrix() const;

    // Change of representation methods
    MatrixXcd to_matrix();

    // Transform methods
    Hamiltonian to_rn_hamiltonian(const size_t t_r_num, const vector<int>& t_signs) const;
//    Hamiltonian to_wider_space(const size_t t_desired_qubit_num) const;
    void to_wider_space(const size_t t_desired_qubit_num);
    Hamiltonian reduce_locality(const size_t t_desired_locality) const;
    IndexPairTerms get_index_pair_terms(const size_t t_desired_locality, const Terms& t_terms) const;
    vector<size_t> partial_ops_to_index_vector(const PartialOps& t_partial_ops) const;
    IndexPair get_max_term_num_index_pair(const IndexPairTerms& t_index_pair_terms) const;
    static void print_index_pair_terms(const IndexPairTerms& t_index_pair_terms);

    // Solution methods
    void solve(solution_method t_solution_method);
    void solve_eigen();
    void solve_brute_force();
    void solve_simcim_python();
    void solve_simcim_python_k_local();
    void solve_simcim_cpp();
    void solve_simcim_cpp_k_local();

    // Information extracting methods
    double ground_state_energy(solution_method t_solution_method);
    VectorXcd ground_state(solution_method t_solution_method);
    void save_ising_matrix_and_vector(const path& t_dir);
    void save_k_tensor_to_sparse_txt(const size_t t_k,
                                     const path& t_dir);
    vector<size_t> get_locality_occ();

    // Overloaded operators for convenience
    friend std::ostream & operator << (std::ostream& t_out, const Hamiltonian& t_hamiltonian);

    // Hamiltonian-Term arithmetics
    friend Hamiltonian operator + (const Hamiltonian& t_hamiltonian, const Term& t_term);
    Hamiltonian& operator += (const Term& t_rhs_term);

    friend Hamiltonian operator - (const Hamiltonian& t_hamiltonian, const Term& t_term);
    Hamiltonian& operator -= (const Term& t_rhs_term);

    //Hamiltonian& operator ^= (const Term& t_rhs_term);

    // Hamiltonian-Hamiltonian arithmetics
    friend Hamiltonian operator + (const Hamiltonian& t_l_hamiltonian, const Hamiltonian& t_r_hamiltonian);
    Hamiltonian& operator += (const Hamiltonian& t_rhs_hamiltonian);

    friend Hamiltonian operator - (const Hamiltonian& t_hamiltonian1, const Hamiltonian& t_hamiltonian2);
    Hamiltonian& operator -= (const Hamiltonian& t_rhs_hamiltonian);

    friend Hamiltonian operator - (const Hamiltonian& t_hamiltonian);

    friend Hamiltonian operator* (const Hamiltonian& t_hamiltonian, const double& t_val);
    friend Hamiltonian operator* (const double& t_val, const Hamiltonian& t_hamiltonian) { return t_hamiltonian * t_val; };
    Hamiltonian& operator *= (const double& t_val);

    friend Hamiltonian operator* (const Hamiltonian& t_l_hamiltonian, const Hamiltonian& t_r_hamiltonian);
    Hamiltonian& operator *= (const Hamiltonian& t_rhs_hamiltonian);

    friend Hamiltonian operator^ (const Hamiltonian& t_hamiltonian, const Term& t_term);
    Hamiltonian& operator ^= (const Term& t_term);

    friend Hamiltonian operator^ (const Hamiltonian& t_l_hamiltonian, const Hamiltonian& t_r_hamiltonian);
    Hamiltonian& operator ^= (const Hamiltonian& t_rhs_hamiltonian);

    friend bool operator == (const Hamiltonian& t_lhs_hamiltonian, const Hamiltonian& t_rhs_hamiltonian);
    friend bool operator != (const Hamiltonian& t_lhs_hamiltonian, const Hamiltonian& t_rhs_hamiltonian) { return !(t_lhs_hamiltonian == t_rhs_hamiltonian); };
};

#endif //SIMCIM_QUANTUM_CHEMISTRY_HAMILTONIAN_H
