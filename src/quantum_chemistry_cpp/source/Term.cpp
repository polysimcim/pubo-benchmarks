 //
// Created by Aleksei Malyshev on 18/10/2019.
//

#include "../headers/Term.h"
#include "../headers/Hamiltonian.h"

const unordered_set<char> Term::PAULI_CHARS = {Term::PAULI_UNITY_CHAR,
                                               Term::PAULI_X_CHAR,
                                               Term::PAULI_Y_CHAR,
                                               Term::PAULI_Z_CHAR};

const Matrix2cd Term::PAULI_UNITY = [] {
    Eigen::Matrix2cd tmp;
    tmp << complex(1.0, 0.0), complex(0.0, 0.0),
           complex(0.0, 0.0), complex(1.0, 0.0);
    return tmp;
}();
const Matrix2cd Term::PAULI_X = [] {
    Eigen::Matrix2cd tmp;
    tmp << complex(0.0, 0.0), complex(1.0, 0.0),
           complex(1.0, 0.0), complex(0.0, 0.0);
    return tmp;
}();
const Matrix2cd Term::PAULI_Y = [] {
    Eigen::Matrix2cd tmp;
    tmp << complex(0.0, 0.0), complex(0.0, -1.0),
           complex(0.0, 1.0), complex(0.0, 0.0);
    return tmp;
}();
const Matrix2cd Term::PAULI_Z = [] {
    Eigen::Matrix2cd tmp;
    tmp << complex(1.0, 0.0), complex(0.0, 0.0),
            complex(0.0, 0.0), complex(-1.0, 0.0);
    return tmp;
}();

const unordered_map<char, Matrix2cd> Term::PAULI_MATRICES = {
        {Term::PAULI_UNITY_CHAR, Term::PAULI_UNITY},
        {Term::PAULI_X_CHAR, Term::PAULI_X},
        {Term::PAULI_Y_CHAR, Term::PAULI_Y},
        {Term::PAULI_Z_CHAR, Term::PAULI_Z},
};

Term::Term(const string& t_pauli_string, const double t_weight)
{
    m_locality = 0;
    for (auto idx = 0; idx < t_pauli_string.size(); idx++)
    {
        assert(PAULI_CHARS.find(t_pauli_string[idx]) != PAULI_CHARS.end());
        if (t_pauli_string[idx] != PAULI_UNITY_CHAR)
        {
            m_partial_ops[idx] = t_pauli_string[idx];
            m_locality++;
        }
    }
    m_qubit_num = t_pauli_string.size();
    m_weight = t_weight;
}

//Term::Term(const size_t t_qubit_num, const double t_weight) : Term(string(t_qubit_num, PAULI_UNITY_CHAR), t_weight)
//{
//
//}

Term::Term(const PartialOps& t_partial_ops, const size_t t_qubit_num, const double t_weight):
m_partial_ops(t_partial_ops),
m_qubit_num(t_qubit_num),
m_locality(t_partial_ops.size()),
m_weight(t_weight)
{
    for (const auto &op: m_partial_ops)
    {
        assert(op.first < m_qubit_num);
        assert(PAULI_CHARS.find(op.second) != PAULI_CHARS.end());
    }
}

Term::Term(const size_t t_qubit_num, const double t_weight) : Term({}, t_qubit_num, t_weight)
{

}

// Auxiliary math method
MatrixXcd Term::kronecker_product(const MatrixXcd& t_lhs, const MatrixXcd& t_rhs)
{
    size_t lhs_rows = t_lhs.rows();
    size_t rhs_rows = t_rhs.rows();
    size_t result_rows = lhs_rows * rhs_rows;

    size_t lhs_cols = t_lhs.cols();
    size_t rhs_cols = t_rhs.cols();
    size_t result_cols = lhs_cols * rhs_cols;

    MatrixXcd result = MatrixXcd::Constant(result_rows, result_cols, 0);

    for (auto lhs_row_idx = 0; lhs_row_idx < lhs_rows; lhs_row_idx++)
    {
        for (auto lhs_col_idx = 0; lhs_col_idx < lhs_cols; lhs_col_idx++)
        {
            for (auto rhs_row_idx = 0; rhs_row_idx < rhs_rows; rhs_row_idx++)
            {
                for (auto rhs_col_idx = 0; rhs_col_idx < rhs_cols; rhs_col_idx++)
                {
                    result(lhs_row_idx * rhs_rows + rhs_row_idx,
                           lhs_col_idx * rhs_cols + rhs_col_idx) = t_lhs(lhs_row_idx, lhs_col_idx) * t_rhs(rhs_row_idx, rhs_col_idx);
                }
            }
        }
    }

    return result;
}

// Change of representation methods
string Term::to_pauli_string() const
{
    string pauli_string = string(m_qubit_num, PAULI_UNITY_CHAR);
    for (const auto &op: m_partial_ops)
    {
        pauli_string[op.first] = op.second;
    }

    return pauli_string;
}

MatrixXcd Term::to_matrix() const
{
    string pauli_string = this->to_pauli_string();
    MatrixXcd result = PAULI_MATRICES.at(pauli_string[0]);
    for (auto op = pauli_string.begin() + 1; op != pauli_string.end(); op++)
    {
        result = kronecker_product(result, PAULI_MATRICES.at(*op));
    }

    return m_weight * result;
}

size_t Term::to_binary() const
{
    size_t binary = 0u;
    string pauli_string = this->to_pauli_string();
    for (const auto& op: pauli_string)
    {
        assert((op == Term::PAULI_UNITY_CHAR) || (op == Term::PAULI_Z_CHAR));
        binary = binary + ((op == Term::PAULI_UNITY_CHAR) ? 0u:1u);
        binary <<= 1u;
    }
    binary >>= 1u;

    return binary;
}

// Transform methods
Hamiltonian Term::to_rn_hamiltonian(const size_t t_r_num, const vector<int>& t_signs) const
{
    assert(t_signs.size() == t_r_num);
    string pauli_string = this->to_pauli_string();
    size_t r_qubit_num = t_r_num * m_qubit_num;
    Hamiltonian result_hamiltonian(r_qubit_num);

    for (auto j = 0; j < t_r_num; j++)
    {
        assert((t_signs[j] == 1) || (t_signs[j] == -1));
        for (auto k = 0; k < t_r_num; k++)
        {
            assert((t_signs[k] == 1) || (t_signs[k] == -1));
            Hamiltonian jk_hamiltonian((Term(r_qubit_num)));
            size_t y_ops_num = 0;
            for (auto idx = 0; idx < m_qubit_num; idx++)
            {
                Hamiltonian ancilla_hamiltonian(r_qubit_num);
                switch (pauli_string[idx])
                {
                    case Term::PAULI_UNITY_CHAR:
                        if (j == k)
                        {
                            ancilla_hamiltonian += Term(r_qubit_num);
                        }
                        else
                        {
                            ancilla_hamiltonian += Term(r_qubit_num, 0.5);
                            ancilla_hamiltonian += Term({{j * m_qubit_num + idx, PAULI_Z_CHAR},
                                                         {k * m_qubit_num + idx, PAULI_Z_CHAR}}, r_qubit_num, 0.5);
                        }
                        break;
                    case Term::PAULI_X_CHAR:
                        if (j == k)
                        {
                            ancilla_hamiltonian = Hamiltonian(Term(r_qubit_num, 0.0));
                            break;
                        }
                        else
                        {
                            ancilla_hamiltonian += Term(r_qubit_num, 0.5);
                            ancilla_hamiltonian -= Term({{j * m_qubit_num + idx, PAULI_Z_CHAR},
                                                         {k * m_qubit_num + idx, PAULI_Z_CHAR}}, r_qubit_num, 0.5);
                        }
                        break;
                    case Term::PAULI_Y_CHAR:
                        if (j == k)
                        {
                            ancilla_hamiltonian = Hamiltonian(Term(r_qubit_num, 0.0));
                            break;
                        }
                        else
                        {
                            ancilla_hamiltonian += Term({{j * m_qubit_num + idx, PAULI_Z_CHAR}}, r_qubit_num, 0.5);
                            ancilla_hamiltonian -= Term({{k * m_qubit_num + idx, PAULI_Z_CHAR}}, r_qubit_num, 0.5);
                        }
                        y_ops_num++;
                        break;
                    case Term::PAULI_Z_CHAR:
                        if (j == k)
                        {
                            ancilla_hamiltonian += Term({{k * m_qubit_num + idx, PAULI_Z_CHAR}}, r_qubit_num);

                        }
                        else
                        {
                            ancilla_hamiltonian += Term({{j * m_qubit_num + idx, PAULI_Z_CHAR}}, r_qubit_num, 0.5);
                            ancilla_hamiltonian += Term({{k * m_qubit_num + idx, PAULI_Z_CHAR}}, r_qubit_num, 0.5);
                        }
                        break;
                    default:
                        exit(-2);
                        break;
                }
                jk_hamiltonian ^= ancilla_hamiltonian;
            }
            assert((y_ops_num % 2) == 0);
            if ((y_ops_num / 2) % 2)
            {
                jk_hamiltonian *= -1;
            }
            jk_hamiltonian *= t_signs[j] * t_signs[k];
            result_hamiltonian += jk_hamiltonian;
        }
    }

    result_hamiltonian *= m_weight;
    return result_hamiltonian;
}

//Term Term::to_wider_space(const size_t t_desired_qubit_num) const
//{
//    assert(t_desired_qubit_num >= m_qubit_num);
//    return Term(m_partial_ops, t_desired_qubit_num,m_weight);
//}

void Term::to_wider_space(const size_t t_desired_qubit_num)
 {
     assert(t_desired_qubit_num >= m_qubit_num);
     m_qubit_num = t_desired_qubit_num;
 }

Hamiltonian Term::reduce_locality_by_one(size_t t_new_qubit_num) const
{
    assert(m_locality > 2);

    // new_qubit_num = m_qubit_num + 1;
    Hamiltonian reduced_hamiltonian(t_new_qubit_num);
    PartialOps partial_ops = m_partial_ops;
    auto first_op_it = partial_ops.begin();
    auto second_op_it = next(first_op_it);
    auto third_op_it = next(second_op_it);
    auto end_it = partial_ops.end();

    if (m_weight >= 0)
    {
        reduced_hamiltonian += Term(t_new_qubit_num, 7.0);
        reduced_hamiltonian += Term(PartialOps(third_op_it, end_it), t_new_qubit_num, 1.0);
        reduced_hamiltonian += Term(PartialOps(first_op_it, second_op_it), t_new_qubit_num, -3.0);
        reduced_hamiltonian += Term(PartialOps(second_op_it, third_op_it), t_new_qubit_num, -3.0);
        reduced_hamiltonian += Term({{t_new_qubit_num - 1, PAULI_Z_CHAR}}, t_new_qubit_num, 6.0);

        PartialOps partial_ops_1 = PartialOps(third_op_it, end_it);
        partial_ops_1[t_new_qubit_num - 1] = PAULI_Z_CHAR;
        reduced_hamiltonian += Term(partial_ops_1, t_new_qubit_num, 2.0);

        PartialOps partial_ops_2 = PartialOps(third_op_it, end_it);
        partial_ops_2[(*first_op_it).first] = PAULI_Z_CHAR;
        reduced_hamiltonian += Term(partial_ops_2, t_new_qubit_num, -1.0);

        PartialOps partial_ops_3 = PartialOps(third_op_it, end_it);
        partial_ops_3[(*second_op_it).first] = PAULI_Z_CHAR;
        reduced_hamiltonian += Term(partial_ops_3, t_new_qubit_num, -1.0);

        reduced_hamiltonian += Term({{(*first_op_it).first, PAULI_Z_CHAR}, {t_new_qubit_num - 1, PAULI_Z_CHAR}}, t_new_qubit_num, -4.0);
        reduced_hamiltonian += Term({{(*second_op_it).first, PAULI_Z_CHAR}, {t_new_qubit_num - 1, PAULI_Z_CHAR}}, t_new_qubit_num, -4.0);
        reduced_hamiltonian += Term({{(*first_op_it).first, PAULI_Z_CHAR}, {(*second_op_it).first, PAULI_Z_CHAR}}, t_new_qubit_num, 1.0);
        reduced_hamiltonian *= m_weight;
    }
    else
    {
        reduced_hamiltonian += Term(t_new_qubit_num, 5.0);
        reduced_hamiltonian += Term(PartialOps(third_op_it, end_it), t_new_qubit_num, -1.0);
        reduced_hamiltonian += Term(PartialOps(first_op_it, second_op_it), t_new_qubit_num, -1.0);
        reduced_hamiltonian += Term(PartialOps(second_op_it, third_op_it), t_new_qubit_num, -1.0);
        reduced_hamiltonian += Term({{t_new_qubit_num - 1, PAULI_Z_CHAR}}, t_new_qubit_num, 2.0);

        PartialOps partial_ops_1 = PartialOps(third_op_it, end_it);
        partial_ops_1[t_new_qubit_num - 1] = PAULI_Z_CHAR;
        reduced_hamiltonian += Term(partial_ops_1, t_new_qubit_num, -2.0);

        PartialOps partial_ops_2 = PartialOps(third_op_it, end_it);
        partial_ops_2[(*first_op_it).first] = PAULI_Z_CHAR;
        reduced_hamiltonian += Term(partial_ops_2, t_new_qubit_num, 1.0);

        PartialOps partial_ops_3 = PartialOps(third_op_it, end_it);
        partial_ops_3[(*second_op_it).first] = PAULI_Z_CHAR;
        reduced_hamiltonian += Term(partial_ops_3, t_new_qubit_num, 1.0);

        reduced_hamiltonian += Term({{(*first_op_it).first, PAULI_Z_CHAR}, {t_new_qubit_num - 1, PAULI_Z_CHAR}}, t_new_qubit_num, -4.0);
        reduced_hamiltonian += Term({{(*second_op_it).first, PAULI_Z_CHAR}, {t_new_qubit_num - 1, PAULI_Z_CHAR}}, t_new_qubit_num, -4.0);
        reduced_hamiltonian += Term({{(*first_op_it).first, PAULI_Z_CHAR}, {(*second_op_it).first, PAULI_Z_CHAR}}, t_new_qubit_num, 3.0);
        reduced_hamiltonian *= -m_weight;
    }
    return reduced_hamiltonian;
}

 Hamiltonian Term::reduce_locality_by_one(size_t t_idx, size_t t_jdx, size_t t_new_qubit_num) const
 {
     assert(m_locality > 2);
     Hamiltonian reduced_hamiltonian(t_new_qubit_num);

     // Let's find all the operators in the term apart from those we substitute
     PartialOps leftover_partial_ops = this->m_partial_ops;
     const auto& idx_search_result = leftover_partial_ops.find(t_idx);
     leftover_partial_ops.erase(idx_search_result);
     const auto& jdx_search_result = leftover_partial_ops.find(t_jdx);
     leftover_partial_ops.erase(jdx_search_result);

     if (m_weight >= 0)
     {
         reduced_hamiltonian += Term(t_new_qubit_num, 7.0);

         reduced_hamiltonian += Term({{t_idx, Term::PAULI_Z_CHAR}}, t_new_qubit_num, -3.0);
         reduced_hamiltonian += Term({{t_jdx, Term::PAULI_Z_CHAR}}, t_new_qubit_num, -3.0);
         reduced_hamiltonian += Term({{t_new_qubit_num - 1, PAULI_Z_CHAR}}, t_new_qubit_num, 6.0);


         reduced_hamiltonian += Term({{t_idx, Term::PAULI_Z_CHAR}, {t_jdx, Term::PAULI_Z_CHAR}}, t_new_qubit_num, 1.0);
         reduced_hamiltonian += Term({{t_idx, Term::PAULI_Z_CHAR}, {t_new_qubit_num - 1, Term::PAULI_Z_CHAR}}, t_new_qubit_num, -4.0);
         reduced_hamiltonian += Term({{t_jdx, Term::PAULI_Z_CHAR}, {t_new_qubit_num - 1, Term::PAULI_Z_CHAR}}, t_new_qubit_num, -4.0);

         reduced_hamiltonian += Term(leftover_partial_ops, t_new_qubit_num, 1.0);

         PartialOps leftover_plus_new = leftover_partial_ops;
         leftover_plus_new[t_new_qubit_num - 1] = Term::PAULI_Z_CHAR;
         reduced_hamiltonian += Term(leftover_plus_new, t_new_qubit_num, 2.0);

         PartialOps leftover_plus_first = leftover_partial_ops;
         leftover_plus_first[t_idx] = Term::PAULI_Z_CHAR;
         reduced_hamiltonian += Term(leftover_plus_first, t_new_qubit_num, -1.0);

         PartialOps leftover_plus_second = leftover_partial_ops;
         leftover_plus_second[t_jdx] = Term::PAULI_Z_CHAR;
         reduced_hamiltonian += Term(leftover_plus_second, t_new_qubit_num, -1.0);

         reduced_hamiltonian *= this->m_weight;
     }
     else
     {
         reduced_hamiltonian += Term(t_new_qubit_num, 5.0);

         reduced_hamiltonian += Term({{t_idx, Term::PAULI_Z_CHAR}}, t_new_qubit_num, -1.0);
         reduced_hamiltonian += Term({{t_jdx, Term::PAULI_Z_CHAR}}, t_new_qubit_num, -1.0);
         reduced_hamiltonian += Term({{t_new_qubit_num - 1, PAULI_Z_CHAR}}, t_new_qubit_num, 2.0);


         reduced_hamiltonian += Term({{t_idx, Term::PAULI_Z_CHAR}, {t_jdx, Term::PAULI_Z_CHAR}}, t_new_qubit_num, 3.0);
         reduced_hamiltonian += Term({{t_idx, Term::PAULI_Z_CHAR}, {t_new_qubit_num - 1, Term::PAULI_Z_CHAR}}, t_new_qubit_num, -4.0);
         reduced_hamiltonian += Term({{t_jdx, Term::PAULI_Z_CHAR}, {t_new_qubit_num - 1, Term::PAULI_Z_CHAR}}, t_new_qubit_num, -4.0);

         reduced_hamiltonian += Term(leftover_partial_ops, t_new_qubit_num, -1.0);

         PartialOps leftover_plus_new = leftover_partial_ops;
         leftover_plus_new[t_new_qubit_num - 1] = Term::PAULI_Z_CHAR;
         reduced_hamiltonian += Term(leftover_plus_new, t_new_qubit_num, -2.0);

         PartialOps leftover_plus_first = leftover_partial_ops;
         leftover_plus_first[t_idx] = Term::PAULI_Z_CHAR;
         reduced_hamiltonian += Term(leftover_plus_first, t_new_qubit_num, 1.0);

         PartialOps leftover_plus_second = leftover_partial_ops;
         leftover_plus_second[t_jdx] = Term::PAULI_Z_CHAR;
         reduced_hamiltonian += Term(leftover_plus_second, t_new_qubit_num, 1.0);

         reduced_hamiltonian *= -this->m_weight;
     }

     return reduced_hamiltonian;
 }

// Overloaded operators for the convenience
std::ostream& operator << (std::ostream& t_out, const Term& t_term)
{
    if (abs(t_term.m_weight) > Term::PRECISION)
    {
        t_out << ((t_term.m_weight > Term::PRECISION) ? "": "- ") << abs(t_term.m_weight);
        PartialOps partial_ops = t_term.m_partial_ops;
        if (!partial_ops.empty())
        {
            t_out << " * ";
        }
        for (const auto& op: partial_ops)
        {
            t_out << op.second << "_" << op.first << " * ";
        }
        if (!partial_ops.empty())
        {
            t_out << "\b\b\b";
        }
    }
    return t_out;
}

Term operator + (const Term& t_l_term, const Term& t_r_term)
{
#ifdef STRICT_ADDITION
    assert(t_l_term.m_qubit_num == t_r_term.m_qubit_num);
#endif
    assert(t_l_term.m_partial_ops.size() == t_r_term.m_partial_ops.size()
           && equal(t_l_term.m_partial_ops.begin(), t_l_term.m_partial_ops.end(), t_r_term.m_partial_ops.begin()));
    assert(t_l_term.m_locality == t_r_term.m_locality);

    return Term(t_l_term.m_partial_ops, t_l_term.m_qubit_num, t_l_term.m_weight + t_r_term.m_weight);
}

Term& Term::operator += (const Term& t_rhs_term)
{
    *this = (*this) + t_rhs_term;
    return *this;
}

Term operator - (const Term& t_l_term, const Term& t_r_term)
{
#ifdef STRICT_ADDITION
    assert(t_l_term.m_qubit_num == t_r_term.m_qubit_num);
#endif
    assert(t_l_term.m_partial_ops.size() == t_r_term.m_partial_ops.size()
           && equal(t_l_term.m_partial_ops.begin(), t_l_term.m_partial_ops.end(), t_r_term.m_partial_ops.begin()));
    assert(t_l_term.m_locality == t_r_term.m_locality);

    return Term(t_l_term.m_partial_ops, t_l_term.m_qubit_num, t_l_term.m_weight - t_r_term.m_weight);
}

Term& Term::operator -= (const Term& t_rhs_term)
{
    *this = (*this) - t_rhs_term;
    return *this;
}

Term operator - (const Term& t_term)
{
    return Term(t_term.m_partial_ops, t_term.m_qubit_num, -t_term.m_weight);
}

Term operator * (const Term& t_term, const double& t_val)
{
    return Term(t_term.m_partial_ops, t_term.m_qubit_num, t_term.m_weight * t_val);
}

Term& Term::operator *= (const double& t_val)
{
    *this = (*this) * t_val;
    return *this;
}

Term operator * (const Term& t_l_term, const Term& t_r_term)
{
    PartialOps result_partial_ops = t_l_term.m_partial_ops;

    for (auto const &r_op: t_r_term.m_partial_ops)
    {
        result_partial_ops[t_l_term.m_qubit_num + r_op.first] = r_op.second;
    }

    return Term(result_partial_ops, t_l_term.m_qubit_num + t_r_term.m_qubit_num, t_l_term.m_weight * t_r_term.m_weight);
}

 Term& Term::operator *= (const Term& t_rhs_term)
 {
     *this = (*this) * t_rhs_term;
     return *this;
 }

Term operator ^ (const Term& t_l_term, const Term& t_r_term)
{
    assert(t_l_term.m_qubit_num == t_r_term.m_qubit_num);
    PartialOps result_partial_ops;
    for (const auto& l_op: t_l_term.m_partial_ops)
    {
        assert(l_op.second == Term::PAULI_Z_CHAR);
        result_partial_ops[l_op.first] = l_op.second;
    }
    for (const auto &r_op: t_r_term.m_partial_ops)
    {
        assert(r_op.second == Term::PAULI_Z_CHAR);

        const auto& l_op_it = result_partial_ops.find(r_op.first);
        if (l_op_it == result_partial_ops.end())
        {
            result_partial_ops[r_op.first] = r_op.second;
        }
        else
        {
            result_partial_ops.erase(r_op.first);
        }
    }

    return Term(result_partial_ops, t_l_term.m_qubit_num, t_l_term.m_weight * t_r_term.m_weight);
}

Term& Term::operator ^= (const Term& t_rhs_term)
{
    *this = (*this) ^ t_rhs_term;
    return *this;
}

bool operator == (const Term& t_l_term, const Term& t_r_term)
{
    return (t_l_term.m_qubit_num == t_r_term.m_qubit_num)
    && (t_l_term.m_partial_ops.size() == t_r_term.m_partial_ops.size()
        && equal(t_l_term.m_partial_ops.begin(), t_l_term.m_partial_ops.end(), t_r_term.m_partial_ops.begin()))
    && (t_l_term.m_locality == t_r_term.m_locality)
    && (abs(t_l_term.m_weight - t_r_term.m_weight) < Term::PRECISION);
}