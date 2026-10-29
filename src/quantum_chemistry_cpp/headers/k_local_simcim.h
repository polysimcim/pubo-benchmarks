//
// Created by malyshev on 24/12/2019.
//

#ifndef SIMCIM_QUANTUM_CHEMISTRY_K_LOCAL_SIMCIM_H
#define SIMCIM_QUANTUM_CHEMISTRY_K_LOCAL_SIMCIM_H

#include "simcim.h"
#include "Term.h"

class KLocalSimCIM: public SimCIM
{
private:
    size_t m_locality = 0;
    vector<double> m_factorials;

    Terms m_terms;

    double m_constant;

    double m_tensors_norm = 0.0;
    double m_tensors_scale = 0.0;

public:
    KLocalSimCIM(size_t t_spins_num,
                 size_t t_locality,
                 const Terms& t_terms);

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


#endif //SIMCIM_QUANTUM_CHEMISTRY_K_LOCAL_SIMCIM_H
