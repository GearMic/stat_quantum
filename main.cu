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

    bool alt_potential;
};

// const size_t max_threads_per_block = 512;
const size_t max_threads_per_block = 1024;


//// helper functions
__global__
void setup_randomize(curandState_t* state, size_t len, unsigned long long seed)
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x; // TODO: is this correct?
    size_t stride = blockDim.x;

    for (unsigned int i=id; i<len; i+=stride) {
        curand_init(seed, i, 0, &state[i]);
    };
}

__global__
void randomize_double_array(double* array, size_t len, double lower, double upper, curandState_t* state)
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x; // TODO: is this correct?
    size_t stride = blockDim.x;
    curandState_t localState;

    for (unsigned int i=id; i<len; i+=stride) {
        localState = state[i];
        array[i] = lower + (upper - lower) * curand_uniform_double(&localState);
        state[i] = localState;
    };

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
    return 1./2. * params.mu_sq * pow(x, 2) + params.lambda * pow(x, 4); // anharmonic oscillator potential
}

__device__
double action_point(double x0, double x1, metropolis_parameters params)
{
    return params.a * (1./2. * params.m0 * pow((x1-x0), 2) / pow(params.a, 2) + potential(x0, params));
}

__device__
double action_2p(double xm1, double x0, double x1, metropolis_parameters parameters)
{
    double action_0 = action_point(xm1, x0, parameters);
    double action_m1 = action_point(x0, x1, parameters);
    return action_0 + action_m1;
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
void metropolis_step(double* xj, size_t start_offset, metropolis_parameters params, curandState_t* random_state) 
{ // TODO: try only making changes at the very end
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t offset = start_offset + idx * params.metropolis_offset;
    if (offset >= params.N) { // do nothing if the point would be out of range
        return;
    } 
    // xj = xj + offset;

    // curandState_t localState = random_state[idx]; // TODO: have different states for every lattice point?
    curandState_t localState = random_state[offset];

    double xjp = curand_uniform_double(&localState) * (2*params.Delta) - params.Delta + xj[offset];
    double lneighbor = xj[offset - 1];
    double rneighbor = xj[(offset + 1) % params.N];
    double S_delta = action_2p(lneighbor, xjp, rneighbor, params) - action_2p(lneighbor, xj[offset], rneighbor, params);

    // if ((offset+1)%params.N != offset+1) printf("yes %i\n", offset);

    //// debugging
    // printf("xj: %f %f\n", *xj, xjp);
    // double xj_old = *xj;

    if (S_delta <= 0) {
        xj[offset] = xjp;
    }
    else {
        double r = curand_uniform_double(&localState);
        // printf("%f\n", r);
        if (exp(-S_delta) > r) {
            xj[offset] = xjp;
        };
    };

    //// debugging
    // printf("xjnew: %f %f\n", xj_old, *xj);

    // random_state[idx] = localState;
    random_state[offset] = localState;
}

void metropolis_call(metropolis_parameters params, double* x, curandState* random_state, size_t metropolis_blocks, size_t metropolis_kernels) {
    for (size_t start_offset=0; start_offset<params.metropolis_offset; start_offset++) {
        for (size_t o=0; o<params.N_markov; o++) {
            metropolis_step
                <<<metropolis_blocks, metropolis_kernels>>>
                (x, start_offset, params, random_state);
            CUDA_CALL(cudaDeviceSynchronize());
        };
    };
}

void metropolis_algo(metropolis_parameters params, double** ensemble_out, size_t* pitch, size_t* width, size_t* height)
// executes the metropolis algorithm, writes data into ensemble, pitch in bytes into pitch, width in doubles into width, height into height
{
    size_t metropolis_offset = params.metropolis_offset; // offset between kernels. The smaller the number, the more kernels run in parallel. Minimum 2
    size_t N = params.N;

    // determine kernel amounts
    size_t metropolis_kernels = (size_t)ceil( (double)N/metropolis_offset ); // amount of kernels that are run in parallel
    size_t metropolis_blocks = cuda_block_amount(metropolis_kernels, max_threads_per_block);
    size_t threads_per_block = metropolis_kernels;
    if (metropolis_blocks > 1) {
        threads_per_block = max_threads_per_block;
    }
    
    printf("total: %i\tblocks: %i\t perblock: %i\n", metropolis_kernels, metropolis_blocks, threads_per_block);

    // initialize data arrays
    size_t N_measurements = params.N_measure;

    curandState_t *random_state;
    CUDA_CALL(cudaMallocManaged(&random_state, N*sizeof(curandState_t)));
    setup_randomize<<<1, max_threads_per_block>>>(random_state, N, 1234); // NOTE: this could be parallelized more efficiently, but it probably doesn't make a significant difference
    cudaDeviceSynchronize();
    
    double *x, *ensemble;
    CUDA_CALL(cudaMallocManaged(&x, N*sizeof(double)));
    size_t ensemble_pitch;
    CUDA_CALL(cudaMallocPitch(&ensemble, &ensemble_pitch, N*sizeof(double), N_measurements));

    // x[0] = params.x0;
    // x[N] = params.xN;
        
    // metropolis algorithm
    unsigned int measure_index = 0;
    randomize_double_array<<<1, max_threads_per_block>>>(x, N, params.xlower, params.xupper, random_state);
    CUDA_CALL(cudaDeviceSynchronize());

    // wait until equilibrium
    for (size_t j=0; j<params.N_until_equilibrium; j++) {
        metropolis_call(params, x, random_state, metropolis_blocks, threads_per_block);
    }

    // start measuring
    for (size_t j=0; j<params.N_measure; j++) {
        for (size_t k=0; k<params.N_montecarlo; k++) {
            metropolis_call(params, x, random_state, metropolis_blocks, threads_per_block);
        };
        // measure the new lattice configuration
        CUDA_CALL(cudaMemcpy((double*)((char*)ensemble + ensemble_pitch*measure_index), x, N*sizeof(double), cudaMemcpyDeviceToDevice));
        measure_index++;
    };

    // return and cleanup
    *ensemble_out = ensemble;
    *pitch = ensemble_pitch;
    *width = N;
    *height = N_measurements;
    CUDA_CALL(cudaFree(random_state));
    CUDA_CALL(cudaFree(x));
}

void metropolis_allinone(metropolis_parameters params, const char* filename)
{
    double* ensemble;
    size_t pitch, width, height;
    metropolis_algo(params, &ensemble, &pitch, &width, &height);
    export_metropolis_data(filename, ensemble, pitch, width, height);
    CUDA_CALL(cudaFree(ensemble));
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


    metropolis_parameters params = {
    .metropolis_offset = 2,
    .xlower = -2., .xupper = 2., .x0 = 0.0, .xN = 0.0,
    .a = 1., .N = 1000,
    .N_until_equilibrium = 100, .N_lattices = 1, .N_measure = 60, .N_montecarlo = 10, .N_markov = 1, .Delta = 2.0,
    .m0 = 1.0, .lambda = 0.0, .mu_sq = 1.0,
    .f_sq = -1.0, // placeholder value
    .alt_potential = false
    };
    // TODO: remove N_lattices


    // step 1: plot action
    metropolis_parameters params_0 = params;
    params_0.m0 = .5;
    params_0.a = .5;
    params_0.N = 1000; // broken for N>1026?
    params_0.N_lattices = 1;
    params_0.N_until_equilibrium = 0; // called Nt in the paper
    params_0.N_measure = 200;
    params_0.N_montecarlo = 1; //only on 1 for testing purposes
    params_0.N_markov = 5; // called nBar in the paper
    params_0.Delta = 2.0 * sqrt(params_0.a);
    params_0.xlower = -10.;
    params_0.xupper = 10.;

    // params_0.N_measure = 0; // disable this part

    double* ensemble;
    size_t pitch, width, height;
    metropolis_algo(params_0, &ensemble, &pitch, &width, &height);

    double* actions; 
    CUDA_CALL((cudaMallocHost(&actions, height)));

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


    // Fig 4, 5
    metropolis_parameters params_4_5 = params;
    params_4_5.N=1000;
    metropolis_allinone(params_4_5, "harmonic_a.csv");

    // Fig. 6
    metropolis_parameters params_6 = params;
    params_6.m0 = 0.5;
    params_6.N_until_equilibrium = 100;
    params_6.N_measure = 1000;
    params_6.N = 51;
    params_6.N_montecarlo = 20;
    params_6.N_markov = 10;
    params_6.mu_sq = 2.0;
    params_6.lambda = 0.0;
    params_6.a = 0.5;
    params_6.Delta = 2 * sqrt(params_6.a);
    metropolis_allinone(params_6, "harmonic_b.csv");

/*
    // Fig. 7
    metropolis_parameters params_7 = params;
    params_7.N = 50;
    params_7.N_lattices = 1;
    params_7.N_measure = 1;
    params_7.N_montecarlo = 40;
    params_7.N_markov = 5;
    params_7.lambda = 1.0;
    params_7.a = 1.0;
    params_7.Delta = 2 * sqrt(params.a);
    params_7.m0 = 0.5;
//TODO: alt potential
    params_7.f_sq = 0.5;
    metropolis_allinone(params_7, "anharmonic_a.csv");
    params_7.f_sq = 1.0;
    metropolis_allinone(params_7, "anharmonic_b.csv");
    params_7.f_sq = 2.0;
    metropolis_allinone(params_7, "anharmonic_c.csv");

    // Fig. 8
    m0 = 0.5;
    f_sq = 2.0;
    N = 200;
    epsilon = 0.25;
    metropolis_allinone(0., 0., 10, 50, 10, 5, NULL, "anharmonic_e.csv");
    // metropolis_algo(0., 0., 100, 50, 10, 5, NULL, "anharmonic_d.csv");
    // metropolis_algo(0., 0., 100, 50, 1, 5, "anharmonic_d.csv", NULL);

    // //// Fig. 9
    m0 = 0.5;
    f_sq = 2.0;
    N = 303;
    a = 0.25;
    metropolis_allinone(0., 0., 1, 10, 1, 5, NULL, "anharmonic_correlation_a.csv");
    metropolis_allinone(0., 0., 1, 10, 1, 10, NULL, "anharmonic_correlation_b.csv");
    metropolis_allinone(0., 0., 1, 10, 1, 15, NULL, "anharmonic_correlation_c.csv");
*/
} 

// TODO: fix end points (start and end should be regarded as the same point)