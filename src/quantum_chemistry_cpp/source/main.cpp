#include <iostream>
#include <chrono>

#include "../headers/Hamiltonian.h"
#include "../headers/quantum_chemistry.h"
#include "../headers/two_local_simcim.h"
#include "../headers/utils.h"
#include "../headers/use_open_mp.h"

#ifdef USE_OPEN_MP
#include <omp.h>
#endif

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include "../headers/ziggurat.h"

using namespace std;

void print_partial_ops(const PartialOps& t_partial_ops)
{
    for (const auto& op: t_partial_ops)
    {
        cout << op.first << " " << op.second << endl;
    }
}

void print_locality_occ(const vector<size_t>& t_locality_occ)
{
    for (const auto& elem: t_locality_occ)
    {
        cout << elem << " ";
    }
    cout << endl;
}

//int main()
//{
//    Term term("ZZZIZ", -1.0);
//    Hamiltonian hamiltonian(Term("ZIZZI",3.0));
//    hamiltonian += term;
//    Hamiltonian reduced_hamiltonian = hamiltonian.reduce_locality_correct(2);
//    //Hamiltonian reduced_hamiltonian_old = hamiltonian.reduce_locality_old(2);
//
//    cout << hamiltonian << endl;
//    cout << reduced_hamiltonian << endl;
//    //cout << reduced_hamiltonian_old << endl;
//    cout << "Number of qubits in the reduced hamiltonian: " << reduced_hamiltonian.get_qubit_num() << endl;
//
//    cout << "Ground state energy of the PUBO Hamiltonian: " << hamiltonian.ground_state_energy(BRUTE_FORCE) << endl;
//    cout << "Ground state energy of the QUBO Hamiltonian: " << reduced_hamiltonian.ground_state_energy(BRUTE_FORCE) << endl;
//
//    return 0;
//}

int main(int argc, char* argv[])
{
    #ifdef USE_OPEN_MP
    omp_set_dynamic(0);
    omp_set_num_threads(8);
    #endif

//    vector<string> molecules = {"H2_equilibrium_reduced", "H2_equilibrium"};
//    vector<string> molecules = {
//            "LiH_equilibrium_reduced",
//            "LiH_equilibrium"
//    };
    map<string, size_t> molecule_to_begin_r_num = {
            {"H2_equilibrium_reduced", 2},
            {"H2_equilibrium", 2},
            {"LiH_equilibrium_reduced", 5},
            {"LiH_equilibrium", 2}
    };
    map<string, size_t> molecule_to_end_r_num = {
            {"H2_equilibrium_reduced", 10},
            {"H2_equilibrium", 10},
            {"LiH_equilibrium_reduced", 6},
            {"LiH_equilibrium", 6}
    };
    assert(argc == 2);
    string molecule = argv[1];

    cout << "Molecule: " << molecule << endl;
    assert(molecule_to_begin_r_num.find(molecule) != molecule_to_begin_r_num.end());

    #ifdef USE_OPEN_MP
    path umbrella_dir = "../../molecules";
    #else
    path umbrella_dir = "../../../molecules";
    #endif

    path molecule_dir = umbrella_dir / molecule;
    path path_to_hamiltonians = molecule_dir / "hamiltonians";

    path path_to_solutions = molecule_dir / "solutionz";

    ofstream molecule_output("./" + molecule + ".out");
    molecule_output << "r\t";
    molecule_output << "PUBO_qubit_num\t";
    molecule_output << "PUBO_term_num\t";
    molecule_output << "QUBO_qubit_num\t";
    molecule_output << "QUBO_term_num\n";
    molecule_output.flush();

    fs::directory_iterator end_itr;
    for (fs::directory_iterator itr(path_to_hamiltonians); itr != end_itr; itr++)
    {
        path path_to_hamiltonian = itr->path();
        string internuclear_distance = path_to_hamiltonian.leaf().string();

        cout << "R = " << internuclear_distance << endl;
        Hamiltonian hamiltonian = load_hamiltonian_from_pauli_strings(path_to_hamiltonian);

        auto t1 = std::chrono::high_resolution_clock::now();
        for (size_t r_num = molecule_to_begin_r_num[molecule]; r_num <= molecule_to_end_r_num[molecule]; r_num++)
        {
            vector<int> signs(r_num, 1);

            Hamiltonian rn_hamiltonian = hamiltonian.to_rn_hamiltonian(r_num, signs);
            cout << "r = " << r_num << endl;
            size_t rn_qubit_num = rn_hamiltonian.get_qubit_num();
            cout << "rn_qubit_num = " << rn_qubit_num << endl;
            molecule_output.flush();
//                cout << "rn ground state energy = " << rn_hamiltonian.ground_state_energy(BRUTE_FORCE) << endl;

            molecule_output << r_num << "\t";
            molecule_output << rn_qubit_num << "\t";
            molecule_output << rn_hamiltonian.get_terms_num() << "\t";

            Hamiltonian two_local_hamiltonian = rn_hamiltonian.reduce_locality(2);
            size_t two_local_qubit_num = two_local_hamiltonian.get_qubit_num();
            cout << "two_local_qubit_num: " << two_local_qubit_num << endl;
//                cout << "two local ground state energy = " << two_local_hamiltonian.ground_state_energy(BRUTE_FORCE) << endl;

            molecule_output << two_local_qubit_num << "\t";
            molecule_output << two_local_hamiltonian.get_terms_num() << "\n";
            molecule_output.flush();
            cout << endl << endl;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
        cout << "Duration: " << duration << endl;
    }
        molecule_output.close();
//    solve_hamil=tonians(path_to_hamiltonians, path_to_solutions, SIMCIM_CPP_K_LOCAL);

    return 0;
}

//minstd_rand random_engine(random_device{}());
//
//boost::mt19937 rng;
//boost::normal_distribution<> nd(0.0, 0.1);
//boost::variate_generator<boost::mt19937&, boost::normal_distribution<>> var_nor(rng, nd);
//
//normal_distribution<double> gaussian_distribution(0.0, 0.1);
//
//uint32_t seed;
//
//float fn[128];
//uint32_t kn[128];
//float wn[128];
//
//float value;
//
//r4_nor_setup(reinterpret_cast<uint32_t *>(kn), fn, wn);
//seed = 7658291;
//
//ofstream out("./norm.txt");
//
//for (size_t idx = 0; idx < 400000; idx++)
//{
////double a = gaussian_distribution(random_engine);
//float a = r4_nor(seed, kn, fn, wn);
//out << a << endl;
//}
//out.close();
