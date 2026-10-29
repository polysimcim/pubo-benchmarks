//
// Created by Aleksei Malyshev on 18/10/2019.
//

#include "../headers/Hamiltonian.h"

// Constructors
Hamiltonian::Hamiltonian(size_t t_qubit_num):
        m_qubit_num(t_qubit_num),
        m_dim(1u << m_qubit_num)
{
    m_solution_methods[EIGEN] = &Hamiltonian::solve_eigen;
    m_solution_methods[BRUTE_FORCE] = &Hamiltonian::solve_brute_force;
    m_solution_methods[SIMCIM_PYTHON] = &Hamiltonian::solve_simcim_python;
    m_solution_methods[SIMCIM_PYTHON_K_LOCAL] = &Hamiltonian::solve_simcim_python_k_local;
    m_solution_methods[SIMCIM_CPP] = &Hamiltonian::solve_simcim_cpp;
    m_solution_methods[SIMCIM_CPP_K_LOCAL] = &Hamiltonian::solve_simcim_cpp_k_local;
}

Hamiltonian::Hamiltonian(const Term &t_term) : Hamiltonian(t_term.get_qubit_num())
{
    *this += t_term;
}

Hamiltonian::Hamiltonian(const string &t_pauli_string, const double t_weight) : Hamiltonian(Term(t_pauli_string, t_weight))
{

}

void Hamiltonian::flush()
{
    this->m_eigenvectors = MatrixXcd();
    this->m_eigenvalues.clear();

    this->m_ground_state_idx = 0xBAD;
    this->m_ground_state = VectorXcd();
    this->m_ground_state_energy = 0xBAD;
    this->m_is_solved = false;
}

// Getters
string Hamiltonian::get_ground_state_qubit_string() const
{
    string helper_str;
    size_t ground_state_idx = m_ground_state_idx;
    for (auto idx = 0; idx < m_qubit_num; idx++)
    {
        helper_str += to_string((ground_state_idx & 1));
        ground_state_idx >>= 1;
    }
    reverse(helper_str.begin(), helper_str.end());

    return helper_str;
}

double Hamiltonian::get_diag_eigenvalue(size_t t_eigenvalue_idx) const
{
//    vector<int> idx_to_sign(m_dim, 0.0);
//    for (auto idx = 0u; idx < m_dim; idx++)
//    {
//        size_t count = 0;
//        size_t idx_copy = idx;
//        while (idx_copy)
//        {
//            if (idx_copy & 1u)
//            {
//                count++;
//            }
//            idx_copy >>= 1u;
//        }
//        idx_to_sign[idx] = (count % 2) ? -1:1;
//    }
//
//    double eigenvalue = 0.0;
//    for (const auto& term: m_terms)
//    {
//        double term_weight = term.get_weight();
//        size_t term_binary = term.to_binary();
//
//        eigenvalue += term_weight * idx_to_sign[term_binary & t_eigenvalue_idx];
//    }

    vector<size_t> bit_idx_to_two_power(m_qubit_num, 0.0);
    for (auto idx = 0u; idx < m_qubit_num; idx++)
    {
        bit_idx_to_two_power[idx] = 1u << idx;
    }

    double eigenvalue = 0.0;
    for (const auto& term: m_terms)
    {
        PartialOps partial_ops = term.get_partial_ops();
        double term_contribution = term.get_weight();
        for (const auto& op: partial_ops)
        {
            assert(op.second == Term::PAULI_Z_CHAR);
            if (t_eigenvalue_idx & (bit_idx_to_two_power[m_qubit_num - op.first - 1]))
            {
                term_contribution *= -1;
            }
        }

        eigenvalue += term_contribution;
    }

    return eigenvalue;
}

vector<double> Hamiltonian::get_diag_eigenvalues() const
{
    vector<int> idx_to_sign(m_dim, 0.0);
    for (auto idx = 0u; idx < m_dim; idx++)
    {
        size_t count = 0;
        size_t idx_copy = idx;
        while (idx_copy)
        {
            if (idx_copy & 1u)
            {
                count++;
            }
            idx_copy >>= 1u;
        }
        idx_to_sign[idx] = (count % 2) ? -1:1;
    }

    vector<double> eigenvalues(m_dim, 0.0);
    for (const auto& term: m_terms)
    {
        double term_weight = term.get_weight();
        size_t term_binary = term.to_binary();

        size_t idx = 0;

        #pragma omp parallel for shared(eigenvalues) private(idx)
        for (idx = 0; idx < m_dim; idx++)
        {
            eigenvalues[idx] += term_weight * idx_to_sign[term_binary & idx];
        }
    }
    return eigenvalues;
}

VectorXd Hamiltonian::get_ising_vector() const
{
    VectorXd result = VectorXd::Zero(m_qubit_num);
    for (const auto& term: m_terms)
    {
        if (term.get_locality() == 1)
        {
            PartialOps partial_ops = term.get_partial_ops();
            assert(partial_ops.size() == 1);

            auto first_z_it = partial_ops.begin();
            assert((*first_z_it).second == Term::PAULI_Z_CHAR);

            result((*first_z_it).first) = -term.get_weight();
        }
    }
    return result;
}

MatrixXd Hamiltonian::get_ising_matrix() const
{
    MatrixXd result = MatrixXd::Zero(m_qubit_num, m_qubit_num);
    for (const auto& term: m_terms)
    {
        if (term.get_locality() == 2)
        {
            PartialOps partial_ops = term.get_partial_ops();
            assert(partial_ops.size() == 2);

            auto first_z_it = partial_ops.begin();
            assert((*first_z_it).second == Term::PAULI_Z_CHAR);
            auto second_z_it = next(first_z_it);
            assert((*second_z_it).second == Term::PAULI_Z_CHAR);

            result((*first_z_it).first, (*second_z_it).first) = -term.get_weight() / 2;
            result((*second_z_it).first, (*first_z_it).first) = -term.get_weight() / 2;
        }
    }
    return result;
}

// Change of representation methods
MatrixXcd Hamiltonian::to_matrix()
{
    MatrixXcd result;
    if (!(m_terms.empty()))
    {
        auto first_term = m_terms.begin();
        result = (*first_term).to_matrix();

        for (auto term = next(first_term); term != m_terms.end(); term++)
        {
            result += (*term).to_matrix();
        }
    }

    return result;
}

//// Transform methods
Hamiltonian Hamiltonian::to_rn_hamiltonian(const size_t t_r_num, const vector<int> &t_signs) const
{
    Hamiltonian result_hamiltonian(t_r_num * m_qubit_num);
    for (auto& term: m_terms)
    {
        result_hamiltonian += term.to_rn_hamiltonian(t_r_num, t_signs);
    }

    return result_hamiltonian;
}

//Hamiltonian Hamiltonian::to_wider_space(const size_t t_desired_qubit_num) const
//{
//    assert(t_desired_qubit_num >= m_qubit_num);
//    Hamiltonian result_hamiltonian(t_desired_qubit_num);
//
//    for (const auto& term: m_terms)
//    {
//        result_hamiltonian += term.to_wider_space(t_desired_qubit_num);
//    }
//
//    return result_hamiltonian;
//}

void Hamiltonian::to_wider_space(const size_t t_desired_qubit_num)
{
    assert(t_desired_qubit_num >= m_qubit_num);

    for (auto& term: m_terms)
    {
        //term.to_wider_space(t_desired_qubit_num);
        (const_cast<Term&>(term)).to_wider_space(t_desired_qubit_num);
    }
    m_qubit_num = t_desired_qubit_num;
    m_dim = 1 << m_qubit_num;

    flush();
}

Hamiltonian Hamiltonian::reduce_locality(const size_t t_desired_locality) const
{
    Hamiltonian reduced_locality_hamiltonian = *this;
    size_t final_qubit_num = reduced_locality_hamiltonian.get_qubit_num();

    Terms cur_terms = reduced_locality_hamiltonian.m_terms;
    IndexPairTerms cur_index_pair_terms = reduced_locality_hamiltonian.get_index_pair_terms(t_desired_locality, cur_terms);

    cout << "Before the start: " << endl;
    cout << "Number of terms: " << reduced_locality_hamiltonian.get_terms_num() << " ";
    cout << "Locality: " << reduced_locality_hamiltonian.get_locality() << endl;
    for (const auto& elem: reduced_locality_hamiltonian.get_locality_occ())
    {
        cout << elem << " ";
    }
    cout << endl << endl;
    size_t iter_idx = 1;
    while (!cur_index_pair_terms.empty())
    {
        auto t1 = std::chrono::high_resolution_clock::now();

        // Looking for a pair of indices which is present in the largest number of monomials
        size_t max_term_num = 0;
        IndexPair max_term_num_pair;
        for (const auto& entry: cur_index_pair_terms)
        {
            if (entry.second.size() > max_term_num)
            {
                max_term_num = entry.second.size();
                max_term_num_pair = entry.first;
            }
        }
        assert(!((max_term_num_pair.first == 0) && (max_term_num_pair.second == 0)));

        final_qubit_num++;
        reduced_locality_hamiltonian.to_wider_space(final_qubit_num);

        const auto& max_term_num_it = cur_index_pair_terms.find(max_term_num_pair);
        assert(max_term_num_it != cur_index_pair_terms.end());

        for (const auto term_it: max_term_num_it->second)
        {
            reduced_locality_hamiltonian -= (*term_it);
            reduced_locality_hamiltonian += term_it->reduce_locality_by_one(max_term_num_pair.first, max_term_num_pair.second, final_qubit_num);
        }
        cur_terms = reduced_locality_hamiltonian.m_terms;
        cur_index_pair_terms = reduced_locality_hamiltonian.get_index_pair_terms(t_desired_locality, cur_terms);

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

        cout << "After the iteration #" << iter_idx << " of the while-loop." << endl;
        cout << "Number of terms: " << reduced_locality_hamiltonian.get_terms_num() << " ";
        cout << "Locality: " << reduced_locality_hamiltonian.get_locality() << endl;
        for (const auto& elem: reduced_locality_hamiltonian.get_locality_occ())
        {
            cout << elem << " ";
        }
        cout << endl;
        cout << "Time spent per iteration: " << duration << endl;
        cout << endl;

        iter_idx++;
    }

    return reduced_locality_hamiltonian;
}

IndexPairTerms Hamiltonian::get_index_pair_terms(const size_t t_desired_locality, const Terms& t_terms) const
{
    IndexPairTerms result;
    for (auto term_it = t_terms.begin(); term_it != t_terms.end(); term_it++)
    {
        if (term_it->get_locality() > t_desired_locality)
        {
            vector<size_t> index_vector = Hamiltonian::partial_ops_to_index_vector(term_it->get_partial_ops());
            for (size_t idx = 0; idx < index_vector.size() - 1; idx++)
            {
                for (size_t jdx = idx + 1; jdx < index_vector.size(); jdx++)
                {
                    IndexPair cur_index_pair = make_pair(index_vector[idx], index_vector[jdx]);
                    if (result.find(cur_index_pair) == result.end())
                    {
                        set<Term const*> cur_terms;
                        result[cur_index_pair] = cur_terms;
                    }
                    result[cur_index_pair].insert(&(*term_it));
                }
            }
        }
    }

    return result;
}

vector<size_t> Hamiltonian::partial_ops_to_index_vector(const PartialOps& t_partial_ops) const
{
    vector<size_t> result;
    result.reserve(t_partial_ops.size());

    for (const auto& op: t_partial_ops)
    {
        result.push_back(op.first);
    }

    return result;
}

IndexPair Hamiltonian::get_max_term_num_index_pair(const IndexPairTerms& t_index_pair_terms) const
{
    size_t max_term_num = 0;
    IndexPair result;
    for (const auto& entry: t_index_pair_terms)
    {
        if (entry.second.size() > max_term_num)
        {
            max_term_num = entry.second.size();
            result = entry.first;
        }
    }

    return result;
}

void Hamiltonian::print_index_pair_terms(const IndexPairTerms& t_index_pair_terms)
{
    for (const auto& it: t_index_pair_terms)
    {
        cout << "Idx = " << it.first.first << ", Jdx = " << it.first.second << endl;
        for (auto cur_term: it.second)
        {
            cout << (*cur_term) << ", ";
        }
        cout << endl;
    }
}

// Solution methods
void Hamiltonian::solve(solution_method t_solution_method)
{
    (this->*m_solution_methods[t_solution_method])();

    m_is_solved = true;
};

void Hamiltonian::solve_eigen()
{
    //Eigen::ComplexEigenSolver<MatrixXcd> ces;
    Eigen::SelfAdjointEigenSolver<MatrixXcd> ces;
    ces.compute(this->to_matrix());

    m_eigenvectors = ces.eigenvectors();
    m_eigenvalues = vector<complex<double>>(ces.eigenvalues().data(), ces.eigenvalues().data() + m_dim);

    size_t ground_state_idx = 0;
    m_ground_state_energy = m_eigenvalues.at(0).real();

    for (auto idx = 1; idx < m_dim; idx++)
    {
        complex<double> eigenval = ces.eigenvalues()[idx];
        // Check that all Hamiltonian eigenvalues are real
        assert(abs(eigenval.imag()) < Term::PRECISION);
        if (eigenval.real() < m_ground_state_energy)
        {
            m_ground_state_energy = eigenval.real();
            ground_state_idx = idx;
        }
    }
    m_ground_state = m_eigenvectors.col(ground_state_idx);
    m_ground_state_idx = m_dim;
    for (auto idx = 0; idx < m_dim; idx++)
    {
        if (abs(m_eigenvectors(idx, ground_state_idx) - 1.0) < Term::PRECISION)
        {
            m_ground_state_idx = idx;
            break;
        }
    }
   // assert(m_ground_state_idx < m_dim);
}

void Hamiltonian::solve_brute_force()
{
    vector<double> eigenvalues = get_diag_eigenvalues();

    double ground_state_energy = eigenvalues.at(0);
    size_t ground_state_idx = 0;
    for (size_t idx = 1; idx < m_dim; idx++)
    {
        if (eigenvalues[idx] < ground_state_energy)
        {
            ground_state_energy = eigenvalues[idx];
            ground_state_idx = idx;
        }
    }
    m_ground_state_idx = ground_state_idx;
    m_ground_state = VectorXcd::Zero(m_dim);
    m_ground_state(ground_state_idx) = complex<double>(1.0, 0.0);
    m_ground_state_energy = ground_state_energy;
}

void Hamiltonian::solve_simcim_python()
{
    Hamiltonian two_local_hamiltonian = (this->get_locality() > 2) ? this->reduce_locality(2):(*this);
    path tmp_dir = "./silly_tmp";
    two_local_hamiltonian.save_ising_matrix_and_vector(tmp_dir);

    path python_script_name = "../simcim.py";
    string command = "python3 ";
    command += python_script_name.string();
    command += " ";
    command += "--path_to_working_dir ";
    command += tmp_dir.string();
    size_t execution_code = system(command.c_str());
    if (execution_code != 0)
    {
        cout << "Wrong execution code of SimCIM.py: " << execution_code << endl;
        exit(-1);
    }

    ifstream ground_energy_file((tmp_dir / "ground_energy.txt").string());
    ground_energy_file >> m_ground_state_energy;
    ground_energy_file.close();

    ifstream ground_state_file((tmp_dir / "ground_state.txt").string());
    m_ground_state_idx = 0;
    for (auto idx = 0; idx < m_qubit_num; idx++)
    {
        size_t bit;
        ground_state_file >> bit;
        m_ground_state_idx += bit;
        m_ground_state_idx <<= (idx == (m_qubit_num - 1)) ? 0:1;
    }
    ground_state_file.close();
    fs::remove_all(tmp_dir);

    m_ground_state_energy = get_diag_eigenvalue(m_ground_state_idx);
    m_ground_state = VectorXcd::Zero(m_dim);
    m_ground_state(m_ground_state_idx) = complex<double>(1.0, 0.0);
}

void Hamiltonian::solve_simcim_python_k_local()
{
    path tmp_dir = "./silly_tmp";

//    size_t desired_locality = this->get_locality();
//    Hamiltonian reduced_hamiltonian = (this->get_locality() > desired_locality) ? this->reduce_locality(desired_locality):(*this);
    for (size_t tensor_rank = 0; tensor_rank <= this->get_locality(); tensor_rank++)
    {
        this->save_k_tensor_to_sparse_txt(tensor_rank, tmp_dir);
    }

    path python_script_name = "../simcim.py";
    string command = "python3 ";
    command += python_script_name.string();
    command += " ";
    command += "--spins_num " + to_string(this->m_qubit_num) + " ";
    command += "--locality " + to_string(this->get_locality()) + " ";
    command += "--path_to_working_dir ";
    command += tmp_dir.string();
    size_t execution_code = system(command.c_str());
    if (execution_code != 0)
    {
        cout << "Wrong execution code of simcim.py: " << execution_code << endl;
        exit(-1);
    }

    ifstream ground_energy_file((tmp_dir / "ground_energy.txt").string());
    ground_energy_file >> m_ground_state_energy;
    ground_energy_file.close();

    ifstream ground_state_file((tmp_dir / "ground_state.txt").string());
    m_ground_state_idx = 0;
    for (auto idx = 0; idx < m_qubit_num; idx++)
    {
        size_t bit;
        ground_state_file >> bit;
        m_ground_state_idx += bit;
        m_ground_state_idx <<= (idx == (m_qubit_num - 1)) ? 0:1;
    }
    ground_state_file.close();
    fs::remove_all(tmp_dir);

    m_ground_state_energy = get_diag_eigenvalue(m_ground_state_idx);
    m_ground_state = VectorXcd::Zero(m_dim);
    m_ground_state(m_ground_state_idx) = complex<double>(1.0, 0.0);
}

void Hamiltonian::solve_simcim_cpp()
{
    Hamiltonian two_local_hamiltonian = (this->get_locality() > 2) ? this->reduce_locality(2):(*this);
    cout << "Two-local hamiltonian number of qubits: " << two_local_hamiltonian.get_qubit_num() << endl;
    cout << "Two-local hamiltonian number of terms: " << two_local_hamiltonian.get_terms_num() << endl;
    exit(-1);

    double ising_constant = two_local_hamiltonian.get_ising_constant();
    VectorXd ising_vector = two_local_hamiltonian.get_ising_vector();
    MatrixXd ising_matrix = two_local_hamiltonian.get_ising_matrix();

    size_t two_local_qubit_num = two_local_hamiltonian.get_qubit_num();
    TwoLocalSimCIM simcim(two_local_qubit_num,
                          ising_constant,
                          ising_vector,
                          ising_matrix);
    Solution solution = simcim.anneal();

    m_ground_state_idx = 0;
    for (auto idx = 0; idx < m_qubit_num; idx++)
    {
        size_t bit;
        m_ground_state_idx += solution.ground_state[idx];
        m_ground_state_idx <<= (idx == (m_qubit_num - 1)) ? 0:1;
    }

    m_ground_state_energy = get_diag_eigenvalue(m_ground_state_idx);
    m_ground_state = VectorXcd::Zero(m_dim);
    m_ground_state(m_ground_state_idx) = complex<double>(1.0, 0.0);
}

void Hamiltonian::solve_simcim_cpp_k_local()
{
//    size_t desired_locality = this->get_locality();
//    Hamiltonian reduced_hamiltonian = (this->get_locality() > desired_locality) ? this->reduce_locality(desired_locality):(*this);
    cout << "K-local hamiltonian number of qubits: " << this->get_qubit_num() << endl;
    cout << "K-local hamiltonian number of terms: " << this->get_terms_num() << endl;

    KLocalSimCIM simcim(this->m_qubit_num,
                        this->get_locality(),
                        this->m_terms);
    Solution solution = simcim.anneal();

    m_ground_state_idx = 0;
    for (auto idx = 0; idx < m_qubit_num; idx++)
    {
        size_t bit;
        m_ground_state_idx += solution.ground_state[idx];
        m_ground_state_idx <<= (idx == (m_qubit_num - 1)) ? 0:1;
    }
    cout << "m_ground_state_idx: " << m_ground_state_idx << endl;
    m_ground_state_energy = get_diag_eigenvalue(m_ground_state_idx);
    //m_ground_state = VectorXcd::Zero(m_dim);
    //m_ground_state(m_ground_state_idx) = complex<double>(1.0, 0.0);
}

// Information extracting methods
double Hamiltonian::ground_state_energy(solution_method t_solution_method)
{
    if (!m_is_solved)
    {
        solve(t_solution_method);
    }

    return m_ground_state_energy;
}

VectorXcd Hamiltonian::ground_state(solution_method t_solution_method)
{
    if (!m_is_solved)
    {
        solve(t_solution_method);
    }

    return m_ground_state;
}

void Hamiltonian::save_ising_matrix_and_vector(const path& t_dir)
{
    assert(this->get_locality() <= 2);
    if (!exists(t_dir))
    {
        fs::create_directories(t_dir);
    }

    ofstream constant_output_file((t_dir / "0.txt").string());
    constant_output_file << get_ising_constant();
    constant_output_file.close();

    ofstream vector_output_file((t_dir / "1.txt").string());
    vector_output_file << get_ising_vector();
    vector_output_file.close();

    ofstream matrix_output_file((t_dir / "2.txt").string());
    matrix_output_file << get_ising_matrix();
    matrix_output_file.close();
}

void Hamiltonian::save_k_tensor_to_sparse_txt(const size_t t_k,
                                              const path& t_dir)
{
    double factorial = 1.0;
    for (size_t number = 2; number <= t_k; number++)
    {
        factorial *= number;
    }
    if (!exists(t_dir))
    {
        fs::create_directories(t_dir);
    }
    path file_name = t_dir / to_string(t_k);
    ofstream output_file(file_name.string() + ".txt");
    size_t terms_count = 0;
    for (const auto& term: m_terms)
    {
        if (term.get_locality() == t_k)
        {
            PartialOps partial_ops = term.get_partial_ops();
            vector<size_t> indices;
            for (const auto& op: partial_ops)
            {
                assert(op.second == Term::PAULI_Z_CHAR);
                indices.push_back(op.first);
            }
            do
            {
                for (const auto idx: indices)
                {
                    output_file << idx << " ";
                }
                output_file << -(term.get_weight() / factorial) << endl;
                //output_file << -(term.get_weight()) << endl;
            }
            while (next_permutation(indices.begin(), indices.end()));
            terms_count++;
        }
    }
    //cout << "There are " << terms_count << " " << t_k << "-local terms in this hamiltonian" << endl;
    output_file.close();
}

// Overloaded operators for convenience
std::ostream& operator << (std::ostream& t_out, const Hamiltonian& t_hamiltonian)
{
    size_t idx = 0;
    for (const auto& term: t_hamiltonian.m_terms)
    {
        if (idx > 0)
        {
            t_out << " ";
        }
        int sign = term.get_sign();
        if (sign == 1)
        {
            if (idx > 0)
            {
                t_out << "+ ";
            }
            t_out << term;
        }
        else
        {
            if (sign == -1)
            {
                t_out << term;
            }
        }
        idx++;
    }
    return t_out;
}

// Hamiltonian-Term arithmetics
Hamiltonian operator+ (const Hamiltonian& t_hamiltonian, const Term& t_term)
{
#ifdef STRICT_ADDITION
    assert(t_hamiltonian.m_qubit_num == t_term.get_qubit_num());
#endif

    Hamiltonian result_hamiltonian(t_hamiltonian.m_qubit_num);
    result_hamiltonian.m_terms = t_hamiltonian.m_terms;

    auto term_in_hamiltonian_it = result_hamiltonian.m_terms.find(t_term);
    if (term_in_hamiltonian_it == result_hamiltonian.m_terms.end())
    {
        if (abs(t_term.get_weight()) > Term::PRECISION)
        {
            result_hamiltonian.m_terms.insert(t_term);
        }
    }
    else
    {
        Term tmp_term = *term_in_hamiltonian_it;
        result_hamiltonian.m_terms.erase(tmp_term);
        Term result_term = tmp_term + t_term;
        if (abs(result_term.get_weight()) > Term::PRECISION)
        {
            result_hamiltonian.m_terms.insert(result_term);
        }
    }

    return result_hamiltonian;
}

Hamiltonian& Hamiltonian::operator += (const Term& t_rhs_term)
{
#ifdef STRICT_ADDITION
    assert(this->m_qubit_num == t_rhs_term.get_qubit_num());
#endif

    auto term_in_hamiltonian_it = this->m_terms.find(t_rhs_term);
    if ((term_in_hamiltonian_it == this->m_terms.end()))
    {
        if ((abs(t_rhs_term.get_weight()) > Term::PRECISION))
        {
            this->m_terms.insert(t_rhs_term);
        }
    }
    else
    {
        Term tmp_term = *term_in_hamiltonian_it;
        this->m_terms.erase(tmp_term);
        Term result_term = tmp_term + t_rhs_term;
        if (abs(result_term.get_weight()) > Term::PRECISION)
        {
            this->m_terms.insert(result_term);
        }
    }
    flush();

    return *this;
}

Hamiltonian operator- (const Hamiltonian& t_hamiltonian, const Term& t_term)
{
    return t_hamiltonian + (-t_term);
}


Hamiltonian& Hamiltonian::operator -= (const Term& t_rhs_term)
{
    *this += (-t_rhs_term);

    return *this;
}

// Hamiltonian-Hamiltonian arithmetics
Hamiltonian operator+ (const Hamiltonian& t_l_hamiltonian, const Hamiltonian& t_r_hamiltonian)
{
#ifdef STRICT_ADDITION
    assert(t_l_hamiltonian.m_qubit_num == t_r_hamiltonian.m_qubit_num);
#endif
    Hamiltonian result_hamiltonian(t_l_hamiltonian.m_qubit_num);
    for (const auto& l_term: t_l_hamiltonian.m_terms)
    {
        result_hamiltonian += l_term;
    }

    for (const auto& r_term: t_r_hamiltonian.m_terms)
    {
        result_hamiltonian += r_term;
    }

    return result_hamiltonian;
}

Hamiltonian& Hamiltonian::operator += (const Hamiltonian& t_rhs_hamiltonian)
{
#ifdef STRICT_ADDITION
    assert(this->m_qubit_num == t_rhs_hamiltonian.m_qubit_num);
#endif

    for (const auto& term: t_rhs_hamiltonian.m_terms)
    {
        *this += term;
    }

    return *this;
}

Hamiltonian operator- (const Hamiltonian& t_l_hamiltonian, const Hamiltonian& t_r_hamiltonian)
{
#ifdef STRICT_ADDITION
    assert(t_l_hamiltonian.m_qubit_num == t_r_hamiltonian.m_qubit_num);
#endif
    Hamiltonian result_hamiltonian(t_l_hamiltonian.m_qubit_num);
    for (const auto& l_term: t_l_hamiltonian.m_terms)
    {
        result_hamiltonian += l_term;
    }

    for (const auto& r_term: t_r_hamiltonian.m_terms)
    {
        result_hamiltonian -= r_term;
    }

    return result_hamiltonian;
}

Hamiltonian& Hamiltonian::operator -= (const Hamiltonian& t_rhs_hamiltonian)
{
#ifdef STRICT_ADDITION
    assert(this->m_qubit_num == t_rhs_hamiltonian.m_qubit_num);
#endif

    for (const auto& term: t_rhs_hamiltonian.m_terms)
    {
        *this -= term;
    }

    return *this;
}

Hamiltonian operator - (const Hamiltonian& t_hamiltonian)
{
    Hamiltonian result_hamiltonian(t_hamiltonian.m_qubit_num);
    for (auto& term: t_hamiltonian.m_terms)
    {
        result_hamiltonian -= term;
    }

    return result_hamiltonian;
}

Hamiltonian operator* (const Hamiltonian& t_hamiltonian, const double& t_val)
{
    Hamiltonian result_hamiltonian(t_hamiltonian.m_qubit_num);
    for (auto const& term: t_hamiltonian.m_terms)
    {
         result_hamiltonian += t_val * term;
    }

    return result_hamiltonian;
}


Hamiltonian& Hamiltonian::operator *= (const double& t_val)
{
    for (auto& term: this->m_terms)
    {
        const_cast<Term&>(term) *= t_val;
    }
    return *this;
}

Hamiltonian operator* (const Hamiltonian& t_l_hamiltonian, const Hamiltonian& t_r_hamiltonian)
{
    assert(t_l_hamiltonian.m_qubit_num == t_r_hamiltonian.m_qubit_num);
    Hamiltonian result_hamiltonian(t_l_hamiltonian.get_qubit_num() + t_r_hamiltonian.get_qubit_num());
    for (const auto& l_term: t_l_hamiltonian.m_terms)
    {
        for (const auto& r_term: t_r_hamiltonian.m_terms)
        {
            result_hamiltonian += l_term * r_term;
        }
    }

    return result_hamiltonian;
}

Hamiltonian& Hamiltonian::operator *= (const Hamiltonian& t_rhs_hamiltonian)
{
    *this = (*this) * t_rhs_hamiltonian;

    return *this;
}

Hamiltonian operator^ (const Hamiltonian& t_hamiltonian, const Term& t_term)
{
    assert(t_hamiltonian.m_qubit_num == t_term.get_qubit_num());
    Hamiltonian result_hamiltonian(t_hamiltonian.m_qubit_num);
    for (const auto& l_term: t_hamiltonian.m_terms)
    {
        result_hamiltonian += l_term ^ t_term;
    }

    return result_hamiltonian;
}

Hamiltonian& Hamiltonian::operator^= (const Term& t_term)
{
    *this = (*this) ^ t_term;

    return *this;
}

Hamiltonian operator^ (const Hamiltonian& t_l_hamiltonian, const Hamiltonian& t_r_hamiltonian)
{
    assert(t_l_hamiltonian.m_qubit_num == t_r_hamiltonian.m_qubit_num);
    Hamiltonian result_hamiltonian(t_l_hamiltonian.get_qubit_num());
    for (const auto& l_term: t_l_hamiltonian.m_terms)
    {
        for (const auto& r_term: t_r_hamiltonian.m_terms)
        {
            result_hamiltonian += l_term ^ r_term;
        }
    }

    return result_hamiltonian;
}


Hamiltonian& Hamiltonian::operator ^= (const Hamiltonian& t_rhs_hamiltonian)
{
    assert(this->m_qubit_num == t_rhs_hamiltonian.m_qubit_num);
    *this = (*this) ^ t_rhs_hamiltonian;

    return *this;
}

bool operator == (const Hamiltonian& t_lhs_hamiltonian, const Hamiltonian& t_rhs_hamiltonian)
{
    bool are_equal = true;
    if (t_lhs_hamiltonian.m_terms.size() != t_rhs_hamiltonian.m_terms.size())
    {
        are_equal = false;
    }
    else
    {
        for (const auto& rhs_term: t_rhs_hamiltonian.m_terms)
        {
            auto lhs_term_it = t_lhs_hamiltonian.m_terms.find(rhs_term);

            if (lhs_term_it == t_lhs_hamiltonian.m_terms.end())
            {
                are_equal = false;
                break;
            }
            else
            {
                if ((*lhs_term_it) != rhs_term)
                {
                    are_equal = false;
                    break;
                }
            }
        }
    }

    return are_equal;
}

vector<size_t> Hamiltonian::get_locality_occ()
{
    size_t m_locality = this->get_locality();
    vector<size_t> locality_occ(m_locality + 1, 0);
    for (const auto& term: m_terms)
    {
         locality_occ[term.get_locality()]++;
    }

    return locality_occ;
}
