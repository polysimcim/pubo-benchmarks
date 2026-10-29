//
// Created by malyshev on 24/12/2019.
//

#ifndef SIMCIM_QUANTUM_CHEMISTRY_SIMCIM_H
#define SIMCIM_QUANTUM_CHEMISTRY_SIMCIM_H

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <utility>
#include <math.h>
#include <random>

#include <boost/filesystem.hpp>

#include "ziggurat.h"

namespace fs = boost::filesystem;
using fs::path;

#include <Eigen/Dense>

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;

typedef function<double(const double)> PumpBaseFunction;
typedef function<double(const double)> PumpFunction;
typedef function<double(void)> Noise;
struct Solution
{
    double ground_state_energy = 0.0;
    vector<size_t> ground_state;
};

enum PumpFunctions{ARCTAN, TANH};

class SimCIM
{
protected:
    size_t m_spins_num;

    // Variables for fast normal distribution generation
    uint32_t m_kn[128];
    float m_fn[128];
    float m_wn[128];

    uint32_t m_seed;

public:
    const static map<PumpFunctions, PumpBaseFunction> pump_base_functions;

    SimCIM(size_t t_spins_num);

    virtual PumpFunction paper_prepare_pump_function(PumpFunctions t_pump_function,
                                             double t_x_stretch,
                                             double t_y_stretch,
                                             double t_y_displ,
                                             size_t t_run_time) const = 0;
    ofstream prepare_output(const path& t_output_path);

    virtual Solution anneal();
    virtual Solution execute_one_run(const size_t t_run_time,
                             const VectorXd& t_pump_values,
                             const double t_zeta,
                             const double t_sigma,
                             const double t_saturation_value,
                             const double t_alpha,
                             ofstream& t_output) = 0;
};


#endif //SIMCIM_QUANTUM_CHEMISTRY_SIMCIM_H
