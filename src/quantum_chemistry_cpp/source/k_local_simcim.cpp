//
// Created by malyshev on 24/12/2019.
//

#include "../headers/k_local_simcim.h"

KLocalSimCIM::KLocalSimCIM(size_t t_spins_num,
                           size_t t_locality,
                           const Terms& t_terms):
                           SimCIM(t_spins_num),
                           m_locality(t_locality),
                           m_terms(t_terms)
{
    cout << "Spins num: " << m_spins_num << endl;
    cout << "Locality: " << m_locality << endl;
    m_factorials.resize(m_locality + 1);
    m_factorials[0] = 1.0;
    for (size_t idx = 1; idx <= m_locality; idx++)
    {
        m_factorials[idx] = idx * m_factorials[idx - 1];
    }
    for (auto& term: m_terms)
    {
//        cout << "Term before: " << term << endl;
        if (term.get_locality() > 0)
        {
            const_cast<Term&>(term) *= -(1.0 / m_factorials[term.get_locality()]);
        } else
        {
            m_constant = term.get_weight();
        }
//        cout << "Term after: " << term << endl;
        if ((term.get_locality() == 1)
        && (abs(term.get_weight()) > m_tensors_norm))
        {
            m_tensors_norm = abs(term.get_weight());
        }
    }
//    cout << "Tensors norm: " << m_tensors_norm << endl;
    for (auto& term: m_terms)
    {
//        cout << "Term before: " << term << endl;
        if (term.get_locality() > 0)
        {
            const_cast<Term&>(term) *= (term.get_locality() / m_tensors_norm);
        }
//        cout << "Term after: " << term << endl;
    }

    vector<double> scale_references((m_locality - 1) * m_spins_num, 0.0);

    if (m_locality >= 2)
    {
        for (auto& term: m_terms)
        {
            size_t term_locality = term.get_locality();
            if (term_locality >= 2)
            {
                for (auto op: term.get_partial_ops())
                {
                    scale_references[m_spins_num * (term_locality - 2) + op.first] += m_factorials[term_locality - 1] * abs(term.get_weight());
                    if (scale_references[m_spins_num * (term_locality - 2) + op.first] > m_tensors_scale)
                    {
                        m_tensors_scale = scale_references[m_spins_num * (term_locality - 2) + op.first];
                    }
                }
            }
        }
    } else
    {
        m_tensors_scale = 1.0;
    }
//    cout << "TENSORS SCALE: " << m_tensors_scale << endl;
//    m_norm = m_ising_vector.cwiseAbs().maxCoeff();
//    m_ising_vector *= (1 / m_norm);
//    m_ising_matrix *= (2 / m_norm);
//
//    m_max_rowwise_sum = (m_ising_matrix.cwiseAbs()).rowwise().sum().maxCoeff();
//    cout << "m_max_rowwise_sum: " << m_max_rowwise_sum << endl;
}

PumpFunction KLocalSimCIM::paper_prepare_pump_function(PumpFunctions t_pump_function,
                                                         double t_x_stretch,
                                                         double t_y_stretch,
                                                         double t_y_displ,
                                                         size_t t_run_time) const
{
    const PumpBaseFunction& pump_base_function = SimCIM::pump_base_functions.at(t_pump_function);

    return [=](double x) {
        return m_tensors_scale * t_y_stretch * pump_base_function(t_x_stretch * ((x / t_run_time) - 0.5)) + t_y_displ;
    };
}

Solution KLocalSimCIM::execute_one_run(const size_t t_run_time,
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
        for (const auto& term: m_terms)
        {
            size_t term_locality = term.get_locality();
            if (term_locality > 0)
            {
                const PartialOps& term_partial_ops = term.get_partial_ops();
                double term_weight = term.get_weight();
                if (term_locality == 1)
                {
                    delta_x[(*(term_partial_ops.begin())).first] += t_zeta * term_weight;
                } else
                {
                    double term_spins_product = 1.0;
                    for (const auto& op: term_partial_ops)
                    {
                        term_spins_product *= x_values[op.first];
                    }
                    for (const auto& op: term_partial_ops)
                    {
                        double cur_spin_spins_product = term_spins_product;
                        if (x_values[op.first] != 0)
                        {
                            cur_spin_spins_product /= x_values[op.first];
                            delta_x[op.first] += t_zeta * term_weight * cur_spin_spins_product * m_factorials[term_locality - 1];
                        }
                    }
                }
            }
        }
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

    solution.ground_state_energy += m_constant;
    for (const auto& term: m_terms)
    {
        size_t term_locality = term.get_locality();
        if (term_locality > 0)
        {
            double term_spins_product = 1.0;
            for (const auto& op: term.get_partial_ops())
            {
                term_spins_product *= s_values[op.first];
            }
            solution.ground_state_energy -= (term.get_weight() * term_spins_product * m_factorials[term_locality]) * (m_tensors_norm / term_locality);
        }
    }

    for (size_t idx = 0; idx < m_spins_num; idx++)
    {
        assert((s_values[idx] == -1) || (s_values[idx] == 1));
        solution.ground_state.push_back((s_values[idx] == -1) ? 1:0);
    }

    return solution;
}