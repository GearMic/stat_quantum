#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <time.h>

#include <curand_kernel.h>


//// macros for error checking
#ifndef NDEBUG
#define CUDA_CALL(x) do { \
cudaError_t err = x; \
if(err != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    printf("%s\n", cudaGetErrorString(err)); \
}} while(0)
#define CURAND_CALL(x) do { \
cudaError_t err = x; \
if(err != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    printf("\t%s\n", cudaGetErrorString(err)); \
}} while(0)

#else
#define CUDA_CALL(x) do { \
    x; \
}} while(0)
#define CURAND_CALL(x) do { \
    x; \
}} while(0)

#endif


//// parameters
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
};

// __device__ double a;
// __device__ double f_sq;
// __device__ double m0 = 1.0;
// __device__ double mu_sq = 1.0;
// __device__ double lambda = 0.0;
// __device__ double epsilon = 1.;
// __device__ double Delta = 2.;

// double xlower = -2.;
// double xupper = 2.;
// size_t N = 2000;
// size_t max_kernels_per_block = 896;
const size_t max_threads_per_block = 512;


//// helper functions
__global__
void setup_randomize(curandState_t* state, size_t len)
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x; // TODO: is this correct?
    size_t stride = blockDim.x;

    for (unsigned int i=id; i<len; i+=stride) {
        curand_init(1234, i, 0, &state[i]);
    };
}

__global__
void randomize_double_array(double* array, size_t len, double lower, double upper, curandState_t* state)
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x; // TODO: is this correct?
    size_t stride = blockDim.x;
    curandState_t localState = state[id];

    for (unsigned int i=id; i<len; i+=stride) {
        array[i] = lower + (upper - lower) * curand_uniform_double(&localState);
    };

    state[id] = localState;
}

void printfl(double x)
{
    printf("%f\n", x);
}

void export_csv_double_1d(FILE* file, double* arr, size_t cols) // TODO: rename cols parameter
{
    for (int col=0; col<cols; col++) {
        fprintf(file, "%f%s", arr[col], (col==cols-1 ? "":","));
    };
    fprintf(file, "\n");
}

void export_csv_double_2d(FILE* file, double* arr, size_t pitch, size_t width, size_t height)
{
    for (int row=0; row<height; row++) {
        export_csv_double_1d(file, (double*)((char*)arr + row*pitch), width);
    };
}

size_t cuda_block_amount(size_t kernels, size_t max_kernels)
{
    return (int)ceil( (double)(kernels) / max_kernels );
}

//// big functions
__device__
double potential(double x, metropolis_parameters params)
{
    return 1./2. * pow(params.mu_sq, 2) * pow(x, 2) + params.lambda * pow(x, 4); // anharmonic oscillator potential
}

// __device__
// double potential_alt(double x)
// {
//     return lambda * pow( pow(x, 2.f) - f_sq, 2.f );
// }

__device__ double (*potential_ptr)(double, metropolis_parameters) = *potential;

__device__
double action_point(double x0, double x1, metropolis_parameters params)
{
    return params.a * (1./2. * params.m0 * pow((x1-x0), 2) / pow(params.a, 2) + (*potential_ptr)(x0, params));
}

__device__
double action_2p(double xm1, double x0, double x1, metropolis_parameters parameters)
{
    double action_0 = action_point(xm1, x0, parameters);
    double action_m1 = action_point(x0, x1, parameters);
    return action_0 + action_m1;
}

__global__ 
void action_latticeconf(double* lattice, metropolis_parameters params, double* action) 
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= params.N) {
        return;
    };

    lattice += idx;
    *action += action_point(lattice[0], lattice[1], params);
}

__global__ 
void action_latticeconf_synchronous(double* lattice, metropolis_parameters params, double* action) 
{
    for (size_t i=0; i<params.N; i++) {
        *action += action_point(lattice[0], lattice[1], params);
        lattice += 1;
    };
}

void export_metropolis_data(const char filename[], double* ensemble, size_t pitch, size_t width, size_t height)
// write metropolis data. Takes in pointer to data on device memory
{
    double* ensemble_host;
    CUDA_CALL(cudaMallocHost(&ensemble_host, height * width*sizeof(double)));
    CUDA_CALL(cudaMemcpy2D(ensemble_host, width*sizeof(double), ensemble, pitch, width*sizeof(double), height, cudaMemcpyDeviceToHost));
    if (filename) {
        FILE* file = fopen(filename, "w");
        export_csv_double_2d(file, ensemble_host, width*sizeof(double), width, height);
        fclose(file);
    }

    CUDA_CALL(cudaFreeHost(ensemble_host));
}

__global__
void metropolis_step(double* xj, size_t n_points, size_t start_offset, metropolis_parameters params, curandState_t* random_state) 
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    size_t offset = id * params.metropolis_offset + start_offset;
    if (offset >= n_points) { // do nothing if the point would be out of range
        return;
    } 
    xj = xj + offset;

    curandState_t localState = random_state[id];

    double xjp = curand_uniform_double(&localState) * (2*params.Delta) + *xj - params.Delta;
    double S_delta = action_2p(xj[-1], xjp, xj[1], params) - action_2p(xj[-1], *xj, xj[1], params);

    if (S_delta < 0) {
        *xj = xjp;
    }
    else {
        if (exp(-S_delta) > curand_uniform_double(&localState)) {
            *xj = xjp;
        };
    };

    random_state[id] = localState;
}

void metropolis_call(metropolis_parameters parameters, double* x, curandState* random_state, size_t metropolis_blocks, size_t metropolis_kernels) {
    for (size_t start_offset=0; start_offset<parameters.metropolis_offset; start_offset++) {
        for (size_t o=0; o<parameters.N_markov; o++) {
            metropolis_step
                <<<metropolis_blocks, metropolis_kernels>>>
                (x+1, parameters.N-1, start_offset, parameters, random_state);
            CUDA_CALL(cudaDeviceSynchronize());
        };
    };
}

void metropolis_algo(metropolis_parameters parameters, double** ensemble_out, size_t* pitch, size_t* width, size_t* height)
// executes the metropolis algorithm, writes data into ensemble, pitch in bytes into pitch, width in doubles into width, height into height
{
    // parameters that are used directly
    size_t metropolis_offset = parameters.metropolis_offset; // offset between kernels. The smaller the number, the more kernels run in parallel. Minimum 2
    double xlower = parameters.xlower;
    double xupper = parameters.xupper;
    double x0 = parameters.x0;
    double xN = parameters.xN;
    size_t N = parameters.N;
    size_t N_until_equilibrium = parameters.N_until_equilibrium;
    size_t N_lattices = parameters.N_lattices;
    size_t N_measure = parameters.N_measure;
    size_t N_montecarlo = parameters.N_montecarlo;

    // determine kernel amounts
    size_t metropolis_kernels = (size_t)ceil( (double)(N-1) / metropolis_offset ); // amount of kernels that are run in parallel
    size_t metropolis_blocks = (size_t)ceil( (double)(metropolis_kernels) / max_threads_per_block );
    if (metropolis_blocks > 1) {
        metropolis_kernels = max_threads_per_block;
    }

    // initialize data arrays
    size_t N_measurements = N_lattices * N_measure;

    curandState_t *random_state, *random_state_algo;
    CUDA_CALL(cudaMallocManaged(&random_state, (N-1) * sizeof(curandState_t)));
    CUDA_CALL(cudaMallocManaged(&random_state_algo, (N-1) * sizeof(curandState_t)));
    setup_randomize<<<1, max_threads_per_block>>>(random_state, N-1); // NOTE: this could be parallelized more efficiently, but it probably doesn't make a significant difference
    setup_randomize<<<1, max_threads_per_block>>>(random_state_algo, metropolis_kernels); // NOTE: this could be parallelized more efficiently, but it probably doesn't make a significant difference
    cudaDeviceSynchronize();
    
    double *x, *ensemble;
    CUDA_CALL(cudaMallocManaged(&x, (N+1) * sizeof(double)));
    size_t ensemble_pitch;
    CUDA_CALL(cudaMallocPitch(&ensemble, &ensemble_pitch, (N+1) * sizeof(double), N_measurements));

    x[0] = x0;
    x[N] = xN;
        
    // metropolis algorithm
    unsigned int measure_index = 0;
    for (int l=0; l<N_lattices; l++) {
        randomize_double_array<<<1, max_threads_per_block>>>(x+1, N-1, xlower, xupper, random_state);
        CUDA_CALL(cudaDeviceSynchronize());

        // wait until equilibrium
        for (size_t j=0; j<N_until_equilibrium; j++) {
            metropolis_call(parameters, x, random_state_algo, metropolis_blocks, metropolis_kernels);
        }

        // start measuring
        for (size_t j=0; j<N_measure; j++) {
            for (size_t k=0; k<N_montecarlo; k++) {
                metropolis_call(parameters, x, random_state_algo, metropolis_blocks, metropolis_kernels);
            };
            // measure the new lattice configuration
            CUDA_CALL(cudaMemcpy((double*)((char*)ensemble + ensemble_pitch*measure_index), x, (N+1)*sizeof(double), cudaMemcpyDeviceToDevice));
            measure_index++;
        };
    };

    // return and cleanup
    *ensemble_out = ensemble;
    *pitch = ensemble_pitch;
    *width = N+1;
    *height = N_measurements;
    CUDA_CALL(cudaFree(random_state));
    CUDA_CALL(cudaFree(random_state_algo));
    CUDA_CALL(cudaFree(x));
}



int main()
{
    // Query CUDA device properties
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %i\n", i);
        printf("Device name: %s\n", prop.name);
        printf("Max threads per block: %i\n", prop.maxThreadsPerBlock);
    }


    metropolis_parameters parameters = {
    .metropolis_offset = 2,
    .xlower = -2., .xupper = 2., .x0 = 0.0, .xN = 0.0,
    .a = 1., .N = 1000,
    .N_until_equilibrium = 100, .N_lattices = 3, .N_measure = 60, .N_montecarlo = 5, .N_markov = 1, .Delta = 2.0,
    .m0 = 1.0, .lambda = 0.0, .mu_sq = 1.0,
    .f_sq = -1.0 // placeholder value
    };

    double* ensemble;
    size_t pitch, width, height;
    metropolis_parameters params_4_5 = parameters;


    // step 1: plot action
    metropolis_parameters params_0 = parameters;
    params_0.m0 = .5;
    params_0.a = .5;
    params_0.N = 100;
    params_0.N_lattices = 1;
    params_0.N_until_equilibrium = 0;
    params_0.N_measure = 400;

    metropolis_algo(params_0, &ensemble, &pitch, &width, &height);

    double* actions; 
    CUDA_CALL((cudaMallocHost(&actions, height)));

    // for (size_t i=0; i<height; i++) {
    //     printfl(actions[i]);
    // }

    size_t n_blocks = cuda_block_amount(params_0.N-1, max_threads_per_block);

    for (size_t i=0; i<height; i++) {
        // action_latticeconf<<<n_blocks, max_threads_per_block>>>((double*)((char*)ensemble + i*pitch), params_0, actions+i);
        action_latticeconf_synchronous<<<1, 1>>>((double*)((char*)ensemble + i*pitch), params_0, actions+i);
    };
    CUDA_CALL(cudaDeviceSynchronize());

    FILE* file_action = fopen("action.csv", "w");
    export_csv_double_1d(file_action, actions, height);
    fclose(file_action);

    CUDA_CALL(cudaFree(ensemble));
    CUDA_CALL(cudaFreeHost(actions));


    // step 2: Fig 4, 5
    metropolis_algo(params_4_5, &ensemble, &pitch, &width, &height);
    export_metropolis_data("harmonic_a.csv", ensemble, pitch, width, height);
    CUDA_CALL(cudaFree(ensemble));

/*
    //// Fig. 6
    metropolis_parameters parameters_6 = parameters;
    parameters_6.N = 51;
    parameters_6.N_montecarlo = 20;
    parameters_6.mu_sq = 2.0;
    parameters_6.a = 0.5;
    parameters_6.Delta = 2 * sqrt(parameters.a);
    metropolis_algo(parameters_6, "harmonic_b.csv");
*/

    // TODO: use the f_sq potential from here on
    // potential_ptr = *potential_alt;

    // Fig. 7
    // metropolis_parameters parameters_7 = parameters;
    // parameters_7.N = 50;
    // parameters_7.N_lattices = 1;
    // parameters_7.N_measure = 1;
    // parameters_7.N_montecarlo = 40;
    // parameters_7.N_markov = 5;
    // parameters_7.lambda = 1.0;
    // parameters_7.a = 1.0;
    // parameters_7.Delta = 2 * sqrt(parameters.a);
    // parameters_7.m0 = 0.5;

    // parameters_7.f_sq = 0.5;
    // metropolis_algo(parameters_7, "anharmonic_a.csv");
    // parameters_7.f_sq = 1.0;
    // metropolis_algo(parameters_7, "anharmonic_b.csv");
    // parameters_7.f_sq = 2.0;
    // metropolis_algo(parameters_7, "anharmonic_c.csv");

    /*
    // Fig. 8
    m0 = 0.5;
    f_sq = 2.0;
    N = 200;
    epsilon = 0.25;
    metropolis_algo(0., 0., 10, 50, 10, 5, NULL, "anharmonic_e.csv");
    // metropolis_algo(0., 0., 100, 50, 10, 5, NULL, "anharmonic_d.csv");
    // metropolis_algo(0., 0., 100, 50, 1, 5, "anharmonic_d.csv", NULL);

    // //// Fig. 9
    m0 = 0.5;
    f_sq = 2.0;
    N = 303;
    a = 0.25;
    metropolis_algo(0., 0., 1, 10, 1, 5, NULL, "anharmonic_correlation_a.csv");
    metropolis_algo(0., 0., 1, 10, 1, 10, NULL, "anharmonic_correlation_b.csv");
    metropolis_algo(0., 0., 1, 10, 1, 15, NULL, "anharmonic_correlation_c.csv");
    */

} 


// TODO: fix end points (start and end should be regarded as the same point)