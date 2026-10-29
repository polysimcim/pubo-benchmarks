//
// Created by Aleksei Malyshev on 29/10/2019.
//

#ifndef SIMCIM_QUANTUM_CHEMISTRY_TWO_LOCAL_SIMCIM_H
#define SIMCIM_QUANTUM_CHEMISTRY_TWO_LOCAL_SIMCIM_H

#include "simcim.h"

class TwoLocalSimCIM: public SimCIM
{
private:
    double m_ising_constant;
    VectorXd m_ising_vector;
    MatrixXd m_ising_matrix;

    double m_norm;
    double m_max_rowwise_sum;

public:

    TwoLocalSimCIM(size_t t_spins_num,
                   double t_ising_constant,
                   const VectorXd& t_ising_vector,
                   const MatrixXd& t_ising_matrix);

    PumpFunction paper_prepare_pump_function(PumpFunctions t_pump_function,
                                             double t_x_stretch,
                                             double t_y_stretch,
                                             double t_y_displ,
                                             size_t t_run_time) const override final;

    Solution execute_one_run(const size_t t_run_time,
                             const VectorXd& t_pump_values,
                             const double t_zeta,
                             const double t_sigma,
                             const double t_saturation_value,
                             const double t_alpha,
                             ofstream& t_output) override final;
};


#endif //SIMCIM_QUANTUM_CHEMISTRY_TWO_LOCAL_SIMCIM_H
