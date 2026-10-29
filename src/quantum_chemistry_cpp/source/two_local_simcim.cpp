//
// Created by Aleksei Malyshev on 29/10/2019.
//

#include "../headers/two_local_simcim.h"

TwoLocalSimCIM::TwoLocalSimCIM(size_t t_spins_num,
                               const double t_ising_constant,
                               const VectorXd &t_ising_vector,
                               const MatrixXd& t_ising_matrix):
SimCIM(t_spins_num),
m_ising_constant(t_ising_constant),
m_ising_vector(t_ising_vector),
m_ising_matrix(t_ising_matrix)
{
    assert(t_spins_num == t_ising_vector.size());
    assert(t_spins_num == t_ising_matrix.rows());
    assert(t_spins_num == t_ising_matrix.cols());

    m_norm = m_ising_vector.cwiseAbs().maxCoeff();
    m_ising_vector *= (1 / m_norm);
    m_ising_matrix *= (2 / m_norm);

    m_max_rowwise_sum = (m_ising_matrix.cwiseAbs()).rowwise().sum().maxCoeff();
    cout << "m_max_rowwise_sum: " << m_max_rowwise_sum << endl;
}

PumpFunction TwoLocalSimCIM::paper_prepare_pump_function(PumpFunctions t_pump_function,
                                                 double t_x_stretch,
                                                 double t_y_stretch,
                                                 double t_y_displ,
                                                 size_t t_run_time) const
{
    const PumpBaseFunction& pump_base_function = SimCIM::pump_base_functions.at(t_pump_function);

    return [=](double x) {
        return m_max_rowwise_sum * t_y_stretch * pump_base_function(t_x_stretch * ((x / t_run_time) - 0.5)) + t_y_displ;
    };
}

Solution TwoLocalSimCIM::execute_one_run(const size_t t_run_time,
                                 const VectorXd &t_pump_values,
                                 const double t_zeta,
                                 const double t_sigma,
                                 const double t_saturation_value,
                                 const double t_alpha,
                                 ofstream& t_output)
{
    VectorXd x_values = VectorXd::Zero(m_spins_num);
    VectorXd delta_x_values = VectorXd::Zero(m_spins_num);

    for (size_t t = 0; t < t_run_time; t++)
    {
        VectorXd delta_x = t_pump_values[t] * x_values;
        delta_x += t_zeta * (m_ising_matrix * x_values + m_ising_vector);
        VectorXd noise_vector = VectorXd::Zero(m_spins_num);
        for (auto idx = 0; idx < m_spins_num; idx++)
        {
            noise_vector[idx] = t_sigma * r4_nor(m_seed, m_kn, m_fn, m_wn);
        }
        delta_x += noise_vector;
        delta_x_values = (1 - t_alpha) * delta_x_values + t_alpha * delta_x;
        x_values += delta_x_values;
        for (auto idx = 0; idx < m_spins_num; idx++)
        {
            if (abs(x_values[idx]) > t_saturation_value)
            {
                x_values[idx] = (x_values[idx] > 0) ? t_saturation_value:(-t_saturation_value);
            }
        }

//        t_output << t << "\t";
//        t_output << pump_function[t] << "\t";
//        for (size_t idx = 0; idx < m_qubit_num; idx++)
//        {
//            t_output << x_values[idx] << "\t";
//        }
//        t_output << endl;
    }

    vector<int> s_values(m_spins_num, 0);
    for (size_t idx = 0; idx < m_spins_num; idx++)
    {
        s_values[idx] = (x_values[idx] > 0) ? 1 : -1;
    }

    Solution solution;
    solution.ground_state_energy = 0.0;

    for (size_t idx = 0; idx < m_spins_num; idx++)
    {
        for (size_t jdx = 0; jdx < m_spins_num; jdx++)
        {
            solution.ground_state_energy -= 0.5 * s_values[idx] * s_values[jdx] * m_ising_matrix(idx, jdx) * m_norm;
        }
        solution.ground_state_energy -= 0.5 * s_values[idx] * m_ising_vector(idx) * m_norm;
    }

    solution.ground_state_energy += m_ising_constant;

    for (size_t idx = 0; idx < m_spins_num; idx++)
    {
        assert((s_values[idx] == -1) || (s_values[idx] == 1));
        solution.ground_state.push_back((s_values[idx] == -1) ? 1:0);
    }

    return solution;
}