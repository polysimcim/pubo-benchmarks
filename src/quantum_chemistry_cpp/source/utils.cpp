//
// Created by malyshev on 08/11/2019.
//

#include "../headers/utils.h"
#include "../headers/quantum_chemistry.h"

Hamiltonian load_hamiltonian_from_pauli_strings(const path& t_path_to_hamiltonian)
{
    ifstream pauli_strings_file(t_path_to_hamiltonian.string());

    size_t qubit_num = 0;

    string line;
    getline(pauli_strings_file, line);
    stringstream helper_ss(line);
    helper_ss >> qubit_num;

    Hamiltonian loaded_hamiltonian(qubit_num);

    while (getline(pauli_strings_file, line))
    {
        string pauli_string = line.substr(0, qubit_num);

        double weight = 0.0;

        stringstream helper_ss(line);
        helper_ss << line.substr(qubit_num + 1);
        helper_ss >> weight;

        loaded_hamiltonian += Term(pauli_string, weight);
    }
    pauli_strings_file.close();

    return loaded_hamiltonian;
}

Hamiltonian load_hydrogen_hamiltonian_from_pauli_strings(const path& t_path_to_hamiltonian)
{
    size_t f_num = 8;
    vector<double> f_vec(f_num, 0.0);

    ifstream pauli_strings_file(t_path_to_hamiltonian.string());

    size_t qubit_num = 0;

    string line;
    getline(pauli_strings_file, line);
    stringstream helper_ss(line);
    helper_ss >> qubit_num;

    while (getline(pauli_strings_file, line))
    {
        string pauli_string = line.substr(0, qubit_num);

        double weight = 0.0;

        stringstream helper_ss(line);
        helper_ss << line.substr(qubit_num + 1);
        helper_ss >> weight;

        if (pauli_string == "IIII")
        {
            f_vec[0] = weight;
        } else if (pauli_string == "ZIII")
        {
            f_vec[1] = weight;
        } else if (pauli_string == "IZII")
        {
            f_vec[2] = weight;
        } else if (pauli_string == "IIZI")
        {
            f_vec[3] = weight;
        } else if (pauli_string == "ZIZI")
        {
            f_vec[4] = weight;
        } else if (pauli_string == "IZIZ")
        {
            f_vec[5] = weight;
        } else if (pauli_string == "XZXI")
        {
            f_vec[6] = weight;
        } else if (pauli_string == "ZZZI")
        {
            f_vec[7] = weight;
        }
    }

    size_t g_num = 5;
    vector<double> g_vec(g_num, 0.0);
    g_vec[0] = f_vec[0];
    g_vec[1] = 2 * f_vec[1];
    g_vec[2] = 2 * f_vec[3];
    g_vec[3] = 2 * (f_vec[4] + f_vec[7]);
    g_vec[4] = 2 * f_vec[6];

    size_t reduced_qubit_num = 2;
    Hamiltonian reduced_hamiltonian(reduced_qubit_num);

    reduced_hamiltonian += Term("II", g_vec[0]);
    reduced_hamiltonian += Term("ZI", g_vec[1]);
    reduced_hamiltonian += Term("IZ", g_vec[2]);
    reduced_hamiltonian += Term("ZZ", g_vec[3]);
    reduced_hamiltonian += Term("XX", g_vec[4]);
    reduced_hamiltonian += Term("YY", g_vec[4]);

    return reduced_hamiltonian;
}


void transform_hydrogen_hamiltonians_to_ising_matrices(const path& t_path_to_hamiltonians, const path& t_path_to_tensors)
{
    fs::directory_iterator end_itr;
    for (fs::directory_iterator itr(t_path_to_hamiltonians); itr != end_itr; itr++)
    {
        path path_to_hamiltonian = itr->path();
        string internuclear_distance = path_to_hamiltonian.leaf().string();

        if (internuclear_distance == "0_54.txt")
        {
            Hamiltonian hydrogen_hamiltonian = load_hydrogen_hamiltonian_from_pauli_strings(path_to_hamiltonian);
            vector<size_t> r_nums = {3};
            for (const auto r_num: r_nums)
            {
                cout << "r = " << r_num << endl;
                vector<int> signs(r_num, 1);
                for (auto idx = 0; idx <= (r_num / 2); idx++)
                {
                    if (idx > 0)
                    {
                        signs[idx - 1] *= -1;
                    }
                    Hamiltonian mapped_hamiltonian = hydrogen_hamiltonian.to_rn_hamiltonian(r_num, signs);
                    Hamiltonian c_matrix = derive_c_matrix(2, r_num, signs);

                    double lambda = 100;
                    Hamiltonian composite_hamiltonian = (mapped_hamiltonian - lambda * c_matrix);
                    Hamiltonian reduced_hamiltonian = composite_hamiltonian.reduce_locality(2);
                    cout << composite_hamiltonian.ground_state_energy(EIGEN) << endl;

                    path path_to_tensors = t_path_to_tensors;
                    path_to_tensors /= internuclear_distance;
                    path_to_tensors /= "r=" + to_string(r_num);
                    string sign_string;
                    for (const auto &sign: signs)
                    {
                        sign_string += to_string(sign) + "_";
                    }
                    path_to_tensors /= sign_string.substr(0, sign_string.size() - 1);

                    reduced_hamiltonian.save_ising_matrix_and_vector(path_to_tensors);
                    ofstream ground_energy_output_file((path_to_tensors / ("ground_energy.txt")).string());
                    ground_energy_output_file << reduced_hamiltonian.ground_state_energy(BRUTE_FORCE);
                    ground_energy_output_file.close();

                    ofstream ground_state_output_file((path_to_tensors / "ground_state.txt").string());
                    ground_state_output_file << reduced_hamiltonian.get_ground_state_qubit_string();
                    ground_state_output_file.close();
                }
            }
        }
    }
}

void transform_hydrogen_hamiltonians_to_ising_tensors(const path& t_path_to_hamiltonians, const path& t_path_to_tensors)
{
    fs::directory_iterator end_itr;
    for (fs::directory_iterator itr(t_path_to_hamiltonians); itr != end_itr; itr++)
    {
        path path_to_hamiltonian = itr->path();
        string internuclear_distance = path_to_hamiltonian.leaf().string();

        //if (internuclear_distance == "0_54.txt")
        {
            Hamiltonian hydrogen_hamiltonian = load_hydrogen_hamiltonian_from_pauli_strings(path_to_hamiltonian);
            vector<size_t> r_nums = {1, 2};
            for (const auto r_num: r_nums)
            {
                cout << "r = " << r_num << endl;
                vector<int> signs(r_num, 1);
                for (auto idx = 0; idx <= (r_num / 2); idx++)
                {
                    if (idx > 0)
                    {
                        signs[idx - 1] *= -1;
                    }
                    Hamiltonian mapped_hamiltonian = hydrogen_hamiltonian.to_rn_hamiltonian(r_num, signs);
                    Hamiltonian c_matrix = derive_c_matrix(2, r_num, signs);

                    double lambda = 100;
                    Hamiltonian composite_hamiltonian = (mapped_hamiltonian - lambda * c_matrix);
                    for (auto locality = 2; locality <= composite_hamiltonian.get_locality(); locality++)
                    {
                        Hamiltonian reduced_hamiltonian = composite_hamiltonian.reduce_locality(locality);
                        cout << composite_hamiltonian.ground_state_energy(EIGEN) << endl;

                        path path_to_tensors = t_path_to_tensors;
                        path_to_tensors /= internuclear_distance;
                        path_to_tensors /= "r=" + to_string(r_num);
                        string sign_string;
                        for (const auto &sign: signs)
                        {
                            sign_string += to_string(sign) + "_";
                        }
                        path_to_tensors /= sign_string.substr(0, sign_string.size() - 1);
                        path_to_tensors /= ("locality=" + to_string(locality));
                        for (auto tensor_rank = 0; tensor_rank <= locality; tensor_rank++)
                        {
                            reduced_hamiltonian.save_k_tensor_to_sparse_txt(tensor_rank, path_to_tensors);
                        }
                        ofstream ground_energy_output_file((path_to_tensors / ("ground_energy.txt")).string());
                        ground_energy_output_file << reduced_hamiltonian.ground_state_energy(BRUTE_FORCE);
                        ground_energy_output_file.close();

                        ofstream ground_state_output_file((path_to_tensors / "ground_state.txt").string());
                        ground_state_output_file << reduced_hamiltonian.get_ground_state_qubit_string();
                        ground_state_output_file.close();
                    }
                }
            }
        }
    }
}

void transform_hamiltonians_to_ising_matrices(const path& t_path_to_hamiltonians, const path& t_path_to_tensors)
{
    fs::directory_iterator end_itr;
    for (fs::directory_iterator itr(t_path_to_hamiltonians); itr != end_itr; itr++)
    {
        path path_to_hamiltonian = itr->path();
        string internuclear_distance = path_to_hamiltonian.leaf().string();

        Hamiltonian hamiltonian = load_hamiltonian_from_pauli_strings(path_to_hamiltonian);
        vector<size_t> r_nums = {1, 2, 3, 4, 5};
        for (const auto r_num: r_nums)
        {
            cout << "r = " << r_num << endl;
            vector<int> signs(r_num, 1);
            for (auto idx = 0; idx <= (r_num / 2); idx++)
            {
                if (idx > 0)
                {
                    signs[idx - 1] *= -1;
                }
                Hamiltonian mapped_hamiltonian = hamiltonian.to_rn_hamiltonian(r_num, signs);
                Hamiltonian c_matrix = derive_c_matrix(2, r_num, signs);

                double lambda = 100;
                Hamiltonian composite_hamiltonian = (mapped_hamiltonian - lambda * c_matrix);
                Hamiltonian reduced_hamiltonian = composite_hamiltonian.reduce_locality(2);
                cout << composite_hamiltonian.ground_state_energy(BRUTE_FORCE) << endl;

                path path_to_tensors = t_path_to_tensors;
                path_to_tensors /= internuclear_distance;
                path_to_tensors /= "r=" + to_string(r_num);
                string sign_string;
                for (const auto &sign: signs)
                {
                    sign_string += to_string(sign) + "_";
                }
                path_to_tensors /= sign_string.substr(0, sign_string.size() - 1);

                reduced_hamiltonian.save_ising_matrix_and_vector(path_to_tensors);
                ofstream ground_energy_output_file((path_to_tensors / ("ground_energy.txt")).string());
                if (reduced_hamiltonian.get_qubit_num() <= SANITY_QUBIT_NUM)
                {
                    ground_energy_output_file << reduced_hamiltonian.ground_state_energy(BRUTE_FORCE);
                } else
                {
                    ground_energy_output_file << composite_hamiltonian.ground_state_energy(BRUTE_FORCE);
                }
                ground_energy_output_file.close();

                ofstream ground_state_output_file((path_to_tensors / "ground_state.txt").string());
                if (reduced_hamiltonian.get_qubit_num() <= SANITY_QUBIT_NUM)
                {
                    ground_state_output_file << reduced_hamiltonian.get_ground_state_qubit_string();
                }
                else
                {
                    ground_state_output_file << composite_hamiltonian.get_ground_state_qubit_string();
                }
                ground_state_output_file.close();
            }
        }
    }
}

void transform_hamiltonians_to_ising_tensors(const path& t_path_to_hamiltonians, const path& t_path_to_tensors)
{
    fs::directory_iterator end_itr;
    for (fs::directory_iterator itr(t_path_to_hamiltonians); itr != end_itr; itr++)
    {
        path path_to_hamiltonian = itr->path();
        string internuclear_distance = path_to_hamiltonian.leaf().string();

        //if (internuclear_distance == "0_54.txt")
        {
            Hamiltonian hamiltonian = load_hamiltonian_from_pauli_strings(path_to_hamiltonian);
            vector<size_t> r_nums = {1, 2};
            for (const auto r_num: r_nums)
            {
                cout << "r = " << r_num << endl;
                vector<int> signs(r_num, 1);
                for (auto idx = 0; idx <= (r_num / 2); idx++)
                {
                    if (idx > 0)
                    {
                        signs[idx - 1] *= -1;
                    }
                    Hamiltonian mapped_hamiltonian = hamiltonian.to_rn_hamiltonian(r_num, signs);
                    Hamiltonian c_matrix = derive_c_matrix_fast(2, r_num, signs);

                    double lambda = 100;
                    Hamiltonian composite_hamiltonian = (mapped_hamiltonian - lambda * c_matrix);
                    cout << "There are " << composite_hamiltonian.get_terms_num() << " terms in the composite hamiltonian" << endl;
                    for (auto locality = composite_hamiltonian.get_locality(); locality <= composite_hamiltonian.get_locality(); locality++)
                    {
                        Hamiltonian reduced_hamiltonian = composite_hamiltonian.reduce_locality(locality);
                        cout << composite_hamiltonian.ground_state_energy(BRUTE_FORCE) << endl;

                        path path_to_tensors = t_path_to_tensors;
                        path_to_tensors /= internuclear_distance;
                        path_to_tensors /= "r=" + to_string(r_num);
                        string sign_string;
                        for (const auto &sign: signs)
                        {
                            sign_string += to_string(sign) + "_";
                        }
                        path_to_tensors /= sign_string.substr(0, sign_string.size() - 1);
                        path_to_tensors /= ("locality=" + to_string(locality));
                        for (auto tensor_rank = 0; tensor_rank <= locality; tensor_rank++)
                        {
                            reduced_hamiltonian.save_k_tensor_to_sparse_txt(tensor_rank, path_to_tensors);
                        }
                        ofstream ground_energy_output_file((path_to_tensors / ("ground_energy.txt")).string());
                        if (reduced_hamiltonian.get_qubit_num() <= SANITY_QUBIT_NUM)
                        {
                            ground_energy_output_file << reduced_hamiltonian.ground_state_energy(BRUTE_FORCE);
                        } else
                        {
                            ground_energy_output_file << composite_hamiltonian.ground_state_energy(BRUTE_FORCE);
                        }
                        ground_energy_output_file.close();

                        ofstream ground_state_output_file((path_to_tensors / "ground_state.txt").string());
                        if (reduced_hamiltonian.get_qubit_num() <= SANITY_QUBIT_NUM)
                        {
                            ground_state_output_file << reduced_hamiltonian.get_ground_state_qubit_string();
                        } else
                        {
                            ground_state_output_file << composite_hamiltonian.get_ground_state_qubit_string();
                        }
                        ground_state_output_file.close();
                    }
                }
            }
        }
    }
}

void solve_hydrogen_hamiltonians(const path& t_path_to_hamiltonians, const path& t_path_to_solutions, const solution_method t_solution_method)
{
    fs::directory_iterator end_itr;
    for (fs::directory_iterator itr(t_path_to_hamiltonians); itr != end_itr; itr++)
    {
        path path_to_hamiltonian = itr->path();
        string internuclear_distance = path_to_hamiltonian.leaf().string();
        //if (internuclear_distance == "4_95.txt")
        {
            cout << "R = " << internuclear_distance << endl;
            Hamiltonian hydrogen_hamiltonian = load_hydrogen_hamiltonian_from_pauli_strings(path_to_hamiltonian);
            double exact_ground_energy = hydrogen_hamiltonian.ground_state_energy(EIGEN);
            cout << "Real ground state energy: " << exact_ground_energy << endl;

            vector<size_t> r_nums = {1, 2, 3};
            for (const auto r_num: r_nums)
            {
                cout << "r = " << r_num << "\t";
                double ground_state_energy = solve_quantum_chemistry(hydrogen_hamiltonian, r_num, t_solution_method).first;
                cout << "E = " << ground_state_energy << endl;

                path path_to_solutions = t_path_to_solutions;
                path_to_solutions /= internuclear_distance;
                path_to_solutions /= "r=" + to_string(r_num);

                if (!exists(path_to_solutions))
                {
                    fs::create_directories(path_to_solutions);
                }
                ofstream ground_energy_output_file((path_to_solutions / ("ground_energy_" + solution_method_to_str[t_solution_method] + ".txt")).string());
                ground_energy_output_file << ground_state_energy;
                ground_energy_output_file.close();

                ofstream exact_ground_energy_output_file((path_to_solutions / "ground_energy_exact.txt").string());
                exact_ground_energy_output_file << exact_ground_energy;
                exact_ground_energy_output_file.close();
            }
        }
        cout << endl;
    }
}

void solve_hamiltonians(const path& t_path_to_hamiltonians, const path& t_path_to_solutions, const solution_method t_solution_method)
{
    fs::directory_iterator end_itr;
    for (fs::directory_iterator itr(t_path_to_hamiltonians); itr != end_itr; itr++)
    {
        path path_to_hamiltonian = itr->path();
        string internuclear_distance = path_to_hamiltonian.leaf().string();

        //if (internuclear_distance == "0.9")
        {
            cout << "R = " << internuclear_distance << endl;
            Hamiltonian hamiltonian = load_hamiltonian_from_pauli_strings(path_to_hamiltonian);
            cout << "Hamiltonian was successfully loaded." << endl;
            cout << "Starting solving the hamiltonian with Eigen." << endl;
            double exact_ground_energy = hamiltonian.ground_state_energy(EIGEN);
            cout << "Hamiltonian was successfully solved!" << endl;
            cout << "Real ground state energy: " << exact_ground_energy << endl;

            path path_to_solutions = t_path_to_solutions;
            path_to_solutions /= internuclear_distance;

            ofstream exact_ground_energy_output_file((path_to_solutions / "EIGEN.txt").string());
            exact_ground_energy_output_file << exact_ground_energy;
            exact_ground_energy_output_file.close();

            ofstream exact_ground_state_output_file((path_to_solutions / "EIGEN_GROUND_STATE.txt").string());
            exact_ground_state_output_file << hamiltonian.ground_state(EIGEN).real();
            exact_ground_state_output_file.close();
            cout << "Solution was saved" << endl;

            vector<size_t> r_nums = {20};
            for (const auto r_num: r_nums)
            {
                //path path_to_solutions = t_path_to_solutions;
                //path_to_solutions /= internuclear_distance;
                // Uncomment if you want to use XYZ->Z mapping
                path cur_path_to_solutions = path_to_solutions/ ("r=" + to_string(r_num));

                if (exists(cur_path_to_solutions / ("ground_energy_" + solution_method_to_str[t_solution_method] + ".txt")))
                {
                    cout << "The solution was already obtained!" << endl;
                    continue;
                } else
                {
                    cout << "r = " << r_num << "\t";
                    pair<double, string> solution = solve_quantum_chemistry(hamiltonian, r_num, t_solution_method);
                    double ground_state_energy = solution.first;
                    string ground_state_string = solution.second;
                    cout << "E = " << ground_state_energy << "\t";
                    cout << "ground state: " << ground_state_string << endl;

                    if (!exists(cur_path_to_solutions))
                    {
                        fs::create_directories(cur_path_to_solutions);
                    }
                    ofstream ground_energy_output_file((cur_path_to_solutions /
                                                        ("ground_energy_" + solution_method_to_str[t_solution_method] +
                                                         ".txt")).string());
                    ground_energy_output_file << ground_state_energy;
                    ground_energy_output_file.close();

                    ofstream ground_state_output_file((cur_path_to_solutions /
                                                        ("ground_state_" + solution_method_to_str[t_solution_method] +
                                                         ".txt")).string());
                    ground_state_output_file << ground_state_string;
                    ground_state_output_file.close();


                    ofstream exact_ground_energy_output_file((cur_path_to_solutions / "ground_energy_exact.txt").string());
                    //exact_ground_energy_output_file << exact_ground_energy;
                    exact_ground_energy_output_file.close();

                    ofstream exact_ground_state_output_file((cur_path_to_solutions / "./ground_state_exact.txt").string());
                    exact_ground_state_output_file << hamiltonian.ground_state(EIGEN).real();
                    exact_ground_state_output_file.close();
                }
            }

            cout << endl;
        }
    }
}