//
// Created by malyshev on 24/12/2019.
//

#include "../headers/simcim.h"

const map<PumpFunctions, PumpBaseFunction> SimCIM::pump_base_functions = {
        {ARCTAN, [] (double x) { return atan(x); }},
        {TANH, [] (double x) { return tanh(x); }},
};

SimCIM::SimCIM(size_t t_spins_num):
        m_spins_num(t_spins_num)
{
    // Initialize structures required for fast normal distribution generation
    r4_nor_setup(m_kn, m_fn, m_wn);
    m_seed = random_device{}();
}

ofstream SimCIM::prepare_output(const path& t_output_path)
{
    ofstream output(t_output_path.string());
    output << "t" << "\t";
    output << "pump" << "\t";
    for (size_t idx = 0; idx < m_spins_num; idx++)
    {
        output << "x" << idx << "\t";
    }
    output << endl;

    return output;
}

Solution SimCIM::anneal()
{
    PumpFunctions pump_function_name = TANH;

    double x_stretch = 3.0;
    double y_displ = 0.0;
    double y_stretch = 0.2;

    size_t run_time = 2000;
    size_t attempt_num = 5000;

    PumpFunction pump_function = paper_prepare_pump_function(pump_function_name,
                                                             x_stretch,
                                                             y_stretch,
                                                             y_displ,
                                                             run_time);

    VectorXd pump_values = VectorXd::Zero(run_time);
    for (size_t t = 0; t < run_time; t++)
    {
        pump_values[t] = pump_function(t);
    }

    double mu = 0.0;
    double sigma = 0.12;

    path output_path = "./output.txt";
    ofstream output = prepare_output(output_path);

    double zeta = 0.1;
    double saturation_value = 1.0;

    double alpha = 0.8;

    Solution solutions[attempt_num];

    size_t attempt_idx = 0;
    #pragma omp parallel for shared(solutions) private(attempt_idx)
    for (attempt_idx = 0; attempt_idx < attempt_num; attempt_idx++)
    {
        solutions[attempt_idx] = execute_one_run(run_time,
                                                 pump_values,
                                                 zeta,
                                                 sigma,
                                                 saturation_value,
                                                 alpha,
                                                 output);
    }

    Solution best_solution = solutions[0];
    for (attempt_idx = 1; attempt_idx < attempt_num; attempt_idx++)
    {
        if (solutions[attempt_idx].ground_state_energy < best_solution.ground_state_energy)
        {
            best_solution = solutions[attempt_idx];
        }
    }
    if (output.is_open())
    {
        output.close();
    }

    cout << "Ground state energy: " << best_solution.ground_state_energy << endl;
    cout << "Ground state: " << endl;
    for (auto bit: best_solution.ground_state)
    {
        cout << bit << " ";
    }
    cout <<  endl;

    return best_solution;
}