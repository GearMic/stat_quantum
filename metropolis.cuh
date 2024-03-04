#include <curand_kernel.h>

const size_t max_threads_per_block = 1024;
const unsigned long long random_seed = 123480;

struct metropolis_parameters
{
    double kernel_offset; // controls how much space is left between lattice sites being updated in parallel. The lower, the more kernels can run in parallel. Minimum 2
    double xlower; // lower and
    double xupper; // upper border for randomization of lattice points
    double x0; // boundary value. currently unused
    double a; // time interval between consecutive lattice sites
    size_t N; // total number of lattice sites
    size_t N_until_equilibrium; // called Nt in the paper 
    size_t N_measure; // total number of measurements taken
    size_t N_montecarlo; // number of monte carlo steps done before the next measurement.
    size_t N_markov; // called N^Bar in the paper
    double Delta; // range for generating updated lattice point in metropolis algorithm

    // potential parameters
    double m0;
    double lambda;
    double mu_sq;
    double f_sq;
    bool alt_potential; // alternative potential is used when this is true
};


__device__ double potential(double x, metropolis_parameters params);
__device__ double action_point(double x0, double x1, metropolis_parameters params);
__device__ double action_2point(double xm1, double x0, double x1, metropolis_parameters parameters);
__global__ void action_latticeconf(double* lattice, metropolis_parameters params, double* action) ;
void export_metropolis_data(const char filename[], double* ensemble, size_t pitch, size_t width, size_t height);
__global__ void metropolis_step(double* xj, size_t start_offset, metropolis_parameters params, curandState_t* random_state) ;
void metropolis_call(metropolis_parameters params, double* x, curandState* random_state, size_t metropolis_blocks, size_t metropolis_kernels);
void metropolis_algo(metropolis_parameters params, double** ensemble_out, size_t* pitch, size_t* width, size_t* height);
void metropolis_allinone(metropolis_parameters params, const char* filename);