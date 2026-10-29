//
// Created by Aleksei Malyshev on 18/10/2019.
//

#ifndef SIMCIM_QUANTUM_CHEMISTRY_TERM_H
#define SIMCIM_QUANTUM_CHEMISTRY_TERM_H

#include <iostream>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <complex>

#include <Eigen/Dense>

//#define STRICT_ADDITION 0

using namespace std;
using Eigen::Matrix2cd;
using Eigen::MatrixXcd;

typedef map<size_t, char> PartialOps;

class Hamiltonian;

class Term
{
private:
    size_t m_qubit_num = 0;
    PartialOps m_partial_ops;
    size_t m_locality = 0;
    mutable double m_weight = 0.0;

public:
    // Precision of the zero difference between terms
    constexpr static const double PRECISION = 1e-11;

    // Definition of Pauli matrices and their corresponding designation
    constexpr static char PAULI_UNITY_CHAR = 'I';
    constexpr static char PAULI_X_CHAR = 'X';
    constexpr static char PAULI_Y_CHAR = 'Y';
    constexpr static char PAULI_Z_CHAR = 'Z';

    const static unordered_set<char> PAULI_CHARS;

    const static Matrix2cd PAULI_UNITY;
    const static Matrix2cd PAULI_X;
    const static Matrix2cd PAULI_Y;
    const static Matrix2cd PAULI_Z;

    const static unordered_map<char, Matrix2cd> PAULI_MATRICES;

    // Constructors and destructors
    Term() = default;
    Term(const PartialOps& t_partial_ops, const size_t t_qubit_num, const double t_weight=1.0);
    Term(const size_t t_qubit_num, const double t_weight=1.0);
    Term(const string& t_pauli_string, const double t_weight=1.0);

    // Getters and setters
    size_t get_qubit_num() const { return m_qubit_num; };
    const PartialOps& get_partial_ops() const { return m_partial_ops; };
    size_t get_locality() const { return m_locality; };
    double get_weight() const { return m_weight; };
    void set_weight(double t_weight) const { m_weight = t_weight; };
    int get_sign() const { return (abs(m_weight) > PRECISION) ? ((m_weight > 0) ? 1 : -1) : 0; };

    // Auxiliary math methods
    static MatrixXcd kronecker_product(const MatrixXcd& t_lhs, const MatrixXcd& t_rhs);

    // Change of representation methods
    string to_pauli_string() const;
    MatrixXcd to_matrix() const;
    size_t to_binary() const;

    // Transform methods
    Hamiltonian to_rn_hamiltonian(const size_t t_r_num, const vector<int>& t_signs) const;
//    Term to_wider_space(const size_t t_desired_qubit_num) const;
    void to_wider_space(const size_t t_desired_qubit_num);
    Hamiltonian reduce_locality_by_one(size_t t_new_qubit_num) const;
    Hamiltonian reduce_locality_by_one(const size_t t_idx, const size_t t_jdx, size_t t_new_qubit_num) const;

    // Overloaded operators for convenience
    friend std::ostream & operator << (std::ostream& t_out, const Term& t_term);

    friend Term operator+ (const Term& t_l_term, const Term& t_r_term);
    Term& operator += (const Term& t_rhs_term);

    friend Term operator- (const Term& t_l_term, const Term& t_r_term);
    Term& operator -= (const Term& t_rhs_term);

    friend Term operator- (const Term& t_term);

    friend Term operator* (const Term& t_term, const double& t_val);
    friend Term operator* (const double& t_val, const Term& t_term) {   return t_term * t_val; };
    Term& operator *= (const double& t_val);

    friend Term operator* (const Term& t_l_term, const Term& t_r_term);
    Term& operator *= (const Term& t_rhs_term);

    friend Term operator ^ (const Term& t_l_term, const Term& t_r_term);
    Term& operator ^= (const Term& t_rhs_term);

    friend bool operator == (const Term& t_l_term, const Term& t_r_term);
    friend bool operator != (const Term& t_lhs_term, const Term& t_rhs_term) { return !(t_lhs_term == t_rhs_term); };
};

struct TermCmp
{
    bool operator() (const Term& t_l_term, const Term& t_r_term) const
    {
        //assert(t_l_term.get_qubit_num() == t_r_term.get_qubit_num());

        const PartialOps& t_l_partial_ops = t_l_term.get_partial_ops();
        const PartialOps& t_r_partial_ops = t_r_term.get_partial_ops();

        if (t_l_partial_ops.size() != t_r_partial_ops.size())
        {
            return t_l_partial_ops.size() < t_r_partial_ops.size();
        }
        else
        {
            if (t_l_partial_ops.empty())
            {
                return false;
            }
            else
            {
                auto t_l_op_it = t_l_partial_ops.begin();
                auto t_r_op_it = t_r_partial_ops.begin();
                while ((t_l_op_it != t_l_partial_ops.end())
                       && (t_r_op_it != t_r_partial_ops.end())
                       && ((*t_l_op_it).first == (*t_r_op_it).first)
                       && ((*t_l_op_it).second == (*t_r_op_it).second))
                {
                    t_l_op_it = next(t_l_op_it);
                    t_r_op_it = next(t_r_op_it);
                }
                assert(((t_l_op_it == t_l_partial_ops.end()) && (t_r_op_it == t_r_partial_ops.end()))
                || ((t_l_op_it != t_l_partial_ops.end()) && (t_r_op_it != t_r_partial_ops.end())));
                if ((t_l_op_it == t_l_partial_ops.end()) && (t_r_op_it == t_r_partial_ops.end()))
                {
                    return false;
                }
                else if ((*t_l_op_it).first != (*t_r_op_it).first)
                {
                    return (*t_l_op_it).first < (*t_r_op_it).first;
                } else if ((*t_l_op_it).second != (*t_r_op_it).second)
                {
                    return (*t_l_op_it).second < (*t_r_op_it).second;
                } else
                {
                    return false;
                }
            }
        }
    }
};

typedef set<Term, TermCmp> Terms;

#endif //SIMCIM_QUANTUM_CHEMISTRY_TERM_H
