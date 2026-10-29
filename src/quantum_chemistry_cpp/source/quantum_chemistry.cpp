//
// Created by Aleksei Malyshev on 23/10/2019.
//

#include <iostream>

#include "../headers/quantum_chemistry.h"
#include "../headers/Hamiltonian.h"

void check_hydrogen_hamiltonian()
{
    double J = 0.1;
    double gamma = 0.0;
    double B = 0.001;

    size_t qubit_num = 2;
    size_t r_num = 2;
    vector<int> signs = {1, 1};

    Hamiltonian init_hamiltonian = Hamiltonian("XX", (-J * (1 + gamma)) / 2)
                                   + Hamiltonian("YY", (-J * (1 - gamma)) / 2)
                                   + Hamiltonian("ZI", -B)
                                   + Hamiltonian("IZ", -B);

    Hamiltonian hamiltonian_1 = Hamiltonian("ZIII")
                                + Hamiltonian("IZII")
                                + Hamiltonian("IIZI")
                                + Hamiltonian("IIIZ");
    hamiltonian_1 *= (-1.5 * B);

    Hamiltonian hamiltonian_2 = Hamiltonian("IIII")
                                - Hamiltonian("IZIZ")
                                - Hamiltonian("ZIZI")
                                + Hamiltonian("ZZZZ");
    hamiltonian_2 *= ((-J * (1 + gamma)) / 4);

    Hamiltonian hamiltonian_3 = Hamiltonian("ZIIZ")
                                - Hamiltonian("ZZII")
                                - Hamiltonian("IIZZ")
                                + Hamiltonian("IZZI");
    hamiltonian_3 *= ((-J * (1 - gamma)) / 4);

    Hamiltonian hamiltonian_4 = Hamiltonian("ZZIZ")
                                + Hamiltonian("IZZZ")
                                + Hamiltonian("ZZZI")
                                + Hamiltonian("ZIZZ");
    hamiltonian_4 *= (-0.5 * B);

    Hamiltonian theor_expanded_hamiltonian = hamiltonian_1 + hamiltonian_2 + hamiltonian_3 + hamiltonian_4;
    Hamiltonian exp_expanded_hamiltonian = init_hamiltonian.to_rn_hamiltonian(r_num, signs);
    Hamiltonian two_local_hamiltonian = exp_expanded_hamiltonian.reduce_locality(2);

    std::cout << "Theory:     " << theor_expanded_hamiltonian << std::endl;
    std::cout << "Experiment: " << exp_expanded_hamiltonian << std::endl;

    cout << ((theor_expanded_hamiltonian == exp_expanded_hamiltonian) ? "Hamiltonians are equal!"
                                                                      : "Hamiltonians are not equal :-(")
         << endl << endl;

    cout << "2-local hamiltonian: " << two_local_hamiltonian << endl;
    cout << endl;

    cout << "Initial hamiltonian ground state energy: " << init_hamiltonian.ground_state_energy(EIGEN) << endl;
    cout << "Z-only hamiltonian ground state energy: " << exp_expanded_hamiltonian.ground_state_energy(BRUTE_FORCE)
         << endl;
    cout << "2-local hamiltonian ground state energy: " << two_local_hamiltonian.ground_state_energy(BRUTE_FORCE)
         << endl;
    cout << endl;

    Hamiltonian c_matrix = derive_c_matrix(qubit_num, r_num, signs);
    double lambda = 100;
    Hamiltonian composite_matrix = exp_expanded_hamiltonian - lambda * c_matrix;
    cout << composite_matrix.to_matrix() << endl;
    Hamiltonian two_local_composite_matrix = composite_matrix.reduce_locality(2);

    cout << "Initial composite matrix: " << composite_matrix << endl;
    cout << "2-local composite matrix: " << two_local_composite_matrix << endl;

    cout << "Initial composite matrix ground state energy: " << composite_matrix.ground_state_energy(BRUTE_FORCE)
         << endl;
    cout << "2-local composite matrix ground state energy: "
         << two_local_composite_matrix.ground_state_energy(BRUTE_FORCE) << endl;
}

Hamiltonian derive_c_matrix(const size_t t_qubit_num, const size_t t_r_num, const vector<int> &t_signs)
{
    assert(t_signs.size() == t_r_num);

    size_t r_qubit_num = t_qubit_num * t_r_num;
    Hamiltonian c_matrix(r_qubit_num);

    unordered_map<size_t, size_t> two_power_log;
    size_t two_power = 1u;
    for (auto idx = 0; idx < t_qubit_num; idx++)
    {
        two_power_log[two_power] = idx;
        two_power <<= 1u;
    }

    size_t size_config_num = 1u << t_qubit_num;
    vector<int> qubit_signs(t_qubit_num, 1);

    for (size_t sign_config = 0; sign_config < size_config_num; sign_config++)
    {
        /// << "Sign config: " << sign_config << endl;
        if (sign_config != 0u)
        {
            size_t bit_to_flip = two_power_log[((sign_config) ^ (sign_config >> 1u)) ^
                                               ((sign_config - 1u) ^ ((sign_config - 1u) >> 1u))];
            qubit_signs[bit_to_flip] *= -1;
        }

        Hamiltonian cur_sign_config_hamiltonian(r_qubit_num);
        for (auto r_idx = 0; r_idx < t_r_num; r_idx++)
        {
            Hamiltonian cur_rn_hamiltonian((Term(r_qubit_num)));
            for (auto qubit_idx = 0; qubit_idx < t_qubit_num; qubit_idx++)
            {
                cur_rn_hamiltonian ^= (Hamiltonian(Term(r_qubit_num, 0.5))
                                       + Hamiltonian(
                        Term({{r_idx * t_qubit_num + qubit_idx, 'Z'}}, r_qubit_num, 0.5 * qubit_signs[qubit_idx])));
            }
            //cout << "Cur rn hamiltonian terms num: " << cur_rn_hamiltonian.get_terms_num() << endl;
            cur_sign_config_hamiltonian += cur_rn_hamiltonian * t_signs[r_idx];
        }
        //cout << cur_sign_config_hamiltonian << endl;
        //cout << "Cur sign config hamiltonian terms num: " << cur_sign_config_hamiltonian.get_terms_num() << endl;
        Hamiltonian cur_sign_config_hamiltonian_copy = cur_sign_config_hamiltonian;
        cur_sign_config_hamiltonian ^= cur_sign_config_hamiltonian_copy;
        //cout << "Cur sign config hamiltonian terms num after squaring: " << cur_sign_config_hamiltonian.get_terms_num() << endl;

        //cout << "C matrix terms num: " << c_matrix.get_terms_num() << endl;
        c_matrix += cur_sign_config_hamiltonian;
    }

    return c_matrix;
}

Hamiltonian derive_c_matrix_fast(const size_t t_qubit_num, const size_t t_r_num, const vector<int> &t_signs)
{
    assert(t_signs.size() == t_r_num);

    size_t r_qubit_num = t_qubit_num * t_r_num;
    Hamiltonian c_matrix = Hamiltonian(Term(r_qubit_num, t_r_num * 1.0));
    double unity_multiplier = 1.0 / (1u << (t_qubit_num - 1));

    unordered_map<size_t, size_t> two_power_log;
    size_t two_power = 1u;
    for (auto idx = 0; idx < t_qubit_num; idx++)
    {
        two_power_log[two_power] = idx;
        two_power <<= 1u;
    }

    for (size_t idx = 0; idx < t_r_num; idx++)
    {
        for (size_t jdx = idx + 1; jdx < t_r_num; jdx++)
        {
            PartialOps cur_partial_ops;
            for (size_t grey_code_idx = 1; grey_code_idx < (1u << t_qubit_num); grey_code_idx++)
            {
                size_t cur_grey_word = grey_code_idx ^ (grey_code_idx >> 1u);
                size_t prev_grey_word = (grey_code_idx - 1) ^ ((grey_code_idx - 1) >> 1u);
                size_t bit_to_flip = cur_grey_word ^ prev_grey_word;

                size_t first_qubit_idx = t_qubit_num * idx + two_power_log[bit_to_flip];
                size_t second_qubit_idx = t_qubit_num * jdx + two_power_log[bit_to_flip];
                if (bit_to_flip & prev_grey_word)
                {
                    assert(cur_partial_ops.find(first_qubit_idx) != cur_partial_ops.end());
                    assert(cur_partial_ops.find(second_qubit_idx) != cur_partial_ops.end());
                    cur_partial_ops.erase(first_qubit_idx);
                    cur_partial_ops.erase(second_qubit_idx);
                }
                else
                {
                    assert(cur_partial_ops.find(first_qubit_idx) == cur_partial_ops.end());
                    assert(cur_partial_ops.find(second_qubit_idx) == cur_partial_ops.end());
                    cur_partial_ops[first_qubit_idx] = Term::PAULI_Z_CHAR;
                    cur_partial_ops[second_qubit_idx] = Term::PAULI_Z_CHAR;
                }
                c_matrix += Term(cur_partial_ops, r_qubit_num, unity_multiplier * t_signs[idx] * t_signs[jdx]);
            }
            c_matrix += Term(r_qubit_num, unity_multiplier * t_signs[idx] * t_signs[jdx]);
        }
    }

    return c_matrix;
}

void check_c_matrix_derivation(size_t t_qubit_num, size_t t_r_num, const vector<int> &t_signs)
{
    Hamiltonian c_matrix = derive_c_matrix(t_qubit_num, t_r_num, t_signs);

    std::cout << c_matrix << std::endl;
    vector<double> eigenvalues = c_matrix.get_diag_eigenvalues();
    for (auto val: eigenvalues)
    {
        cout << val << " ";
    }
    cout << endl;
    std::cout << c_matrix.to_matrix() << std::endl;
}

void check_paper_example()
{
    double J = 0.1;
    double gamma = 0.0;
    double B = 0.001;

    size_t qubit_num = 2;
    size_t r_num = 10;

    Hamiltonian hydrogen_hamiltonian = Hamiltonian("XX", (-J * (1 + gamma)) / 2)
                                       + Hamiltonian("YY", (-J * (1 - gamma)) / 2)
                                       + Hamiltonian("ZI", -B)
                                       + Hamiltonian("IZ", -B);
    cout << "Initial hamiltonian: " << hydrogen_hamiltonian << endl;
    hydrogen_hamiltonian.solve(EIGEN);

    cout << "Its ground state energy via brute force:       " << hydrogen_hamiltonian.ground_state_energy(EIGEN)
         << endl;
    cout << "Its eigenvectors: " << endl;
    cout << hydrogen_hamiltonian.get_eigenvectors() << endl;
    cout << "Its eigenvalues: " << endl;
    for (auto val: hydrogen_hamiltonian.get_eigenvalues())
    {
        cout << val << " ";
    }
    cout << endl;
    cout << "Its ground state energy via quantum chemistry: " << solve_quantum_chemistry(hydrogen_hamiltonian, r_num).first
         << endl;
}

pair<double, string> solve_quantum_chemistry(const Hamiltonian &t_hamiltonian, const size_t t_r_num, const solution_method t_solution_method)
{
    size_t qubit_num = t_hamiltonian.get_qubit_num();
    vector<int> signs(t_r_num, 1);
    double ground_state_energy = BIG_LAMBDA;
    string ground_state_string;

    for (auto idx = 0; idx <= (t_r_num / 2); idx++)
    {
        if (idx > 0)
        {
            signs[idx - 1] *= -1;
        }
        double lambda = BIG_LAMBDA;
        double prev_lambda = BIG_LAMBDA;
        double max_c_element = idx * idx + (t_r_num - idx) * (t_r_num - idx);

        Hamiltonian mapped_hamiltonian = t_hamiltonian.to_rn_hamiltonian(t_r_num, signs);
        cout << "We have found mapped hamiltonian. ";
        cout << "It has " << mapped_hamiltonian.get_terms_num() << " terms and it is ";
        cout << mapped_hamiltonian.get_locality() << "-local" << endl;
        Hamiltonian c_matrix = derive_c_matrix_fast(qubit_num, t_r_num, signs);
        cout << "We also have found C-matrix. ";
        cout << "It has " << c_matrix.get_terms_num() << " terms and it is ";
        cout << c_matrix.get_locality() << "-local" << endl;
        double cur_ground_state_energy = 0.0;
        string cur_ground_state_string;

        cout << "We start iterating over lambdas. " << endl;
        size_t iter_idx = 1;
        do
        {
            cout << "Iteration #" << iter_idx << " started... ";
            prev_lambda = lambda;
            Hamiltonian composite_hamiltonian = (mapped_hamiltonian - lambda * c_matrix);
            composite_hamiltonian.solve(t_solution_method);

            //VectorXcd cur_ground_state = composite_hamiltonian.ground_state(t_solution_method);
            cur_ground_state_energy = composite_hamiltonian.ground_state_energy(t_solution_method);
            cur_ground_state_string = composite_hamiltonian.get_ground_state_qubit_string();
            cout << "Intermediate ground state: " << cur_ground_state_string << endl;

            double sum_b_two = c_matrix.get_diag_eigenvalue(composite_hamiltonian.get_ground_state_idx());
            cout << "sum_b_two for this ground state: " << sum_b_two << endl;
            if (sum_b_two == max_c_element)
            {
                lambda = cur_ground_state_energy / sum_b_two + lambda;
                break;
            }

            if (abs(sum_b_two) < Term::PRECISION)
            {
                break;
            } else
            {
                lambda += cur_ground_state_energy / sum_b_two;
            }
            cout << "Finished! " << endl;
            iter_idx++;
        } while ((cur_ground_state_energy < 0) && (abs(lambda - prev_lambda) > Term::PRECISION));
        if (lambda < ground_state_energy)
        {
            ground_state_energy = lambda;
            ground_state_string = cur_ground_state_string;
        }
    }
    return make_pair(ground_state_energy, ground_state_string);
}
