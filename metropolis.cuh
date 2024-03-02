#include <curand_kernel.h>

const size_t max_threads_per_block = 1024;
const unsigned long long random_seed = 123480;

struct metropolis_parameters
{
    double metropolis_offset;
    double xlower;
    double xupper;
    double x0;
    double xN;
    double a;
    size_t N;
    size_t N_until_equilibrium;
    size_t N_lattices;   
    size_t N_measure;
    size_t N_montecarlo;
    size_t N_markov;
    double Delta;

    double m0;
    double lambda;
    double mu_sq;
    double f_sq;

    bool alt_potential;
};


__device__ double potential(double x, metropolis_parameters params);
__device__ double action_point(double x0, double x1, metropolis_parameters params);
__device__ double action_2p(double xm1, double x0, double x1, metropolis_parameters parameters);
__global__ void action_latticeconf_synchronous(double* lattice, metropolis_parameters params, double* action) ;
void export_metropolis_data(const char filename[], double* ensemble, size_t pitch, size_t width, size_t height);
__global__ void metropolis_step(double* xj, size_t start_offset, metropolis_parameters params, curandState_t* random_state) ;
void metropolis_call(metropolis_parameters params, double* x, curandState* random_state, size_t metropolis_blocks, size_t metropolis_kernels);
void metropolis_algo(metropolis_parameters params, double** ensemble_out, size_t* pitch, size_t* width, size_t* height);
void metropolis_allinone(metropolis_parameters params, const char* filename);