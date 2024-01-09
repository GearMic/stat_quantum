// clang -o a main.c -lm

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <time.h>

#include <curand_kernel.h>

// macro for error checking
#define CUDA_CALL(x) do { \
if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
}} while(0)
#define CURAND_CALL(x) do { \
if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
}} while(0)
    // return EXIT_FAILURE;} 


//// parameters
// __device__ size_t N;
// __device__ double epsilon;
__device__ double a;
// __device__ double Delta;
// __device__ double m0;
// __device__ double mu_sq;
// __device__ double lambda;
__device__ double f_sq;
// __device__ double xlower = -2.;
// __device__ double xupper = 2.;
double xlower = -2.;
double xupper = 2.;


__device__ double m0 = 1.0;
__device__ double mu_sq = 1.0;
__device__ double lambda = 0.0;
size_t N = 1000;
// size_t N = 257; // for testing
__device__ double epsilon = 1.;
__device__ double Delta = 2.;




// // helper functions
// double frand(double lower, double upper) // TODO: do this on the graphics card
// {
//     // static int seed;
//     // seed = rand();
//     // srand(seed);
//     return lower + (upper - lower) * ((double)rand() / (double)RAND_MAX);
// }

__global__
void setup_randomize(curandState_t* state)
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x; // TODO: is this correct?
    curand_init(1234, id, 0, &state[id]);
}

__global__
void randomize_double_array(double* array, size_t len, double lower, double upper, curandState_t* state)
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x; // TODO: is this correct?
    // size_t id = threadIdx.x;
    curandState_t localState = state[id];

    size_t stride = blockDim.x;
    for (unsigned int i=id; i<len; i+=stride) {
        array[i] = lower + (upper - lower) * curand_uniform_double(&localState);
    };

    state[id] = localState;
}

// __global__
// void test_fn(double*array, size_t len)
// {
//     size_t id = threadIdx.x;
//     size_t stride = blockDim.x;
//     for (unsigned int i=id; i<len; i+=stride) {
//         array[i] = 0.5;
//     };
// }


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


// big functions
__device__
double potential(double x)
{
    return 1./2. * pow(mu_sq, 2) * pow(x, 2) + lambda * pow(x, 4); // anharmonic oscillator potential
}

__device__
double potential_alt(double x)
{
    return lambda * pow( pow(x, 2.f) - f_sq, 2.f );
}

__device__ double (*potential_ptr)(double) = *potential;

__device__
double action_point(double x0, double x1)
{
    return epsilon * (1./2. * m0 * pow((x1-x0), 2) / pow(epsilon, 2) + (*potential_ptr)(x0));
}

__device__
double action_2p(double xm1, double x0, double x1)
{
    double action_0 = action_point(xm1, x0);
    double action_m1 = action_point(x0, x1);
    return action_0 + action_m1;
}

__global__
void metropolis_step(double* xj, size_t metropolis_offset, size_t start_offset, double xlower, double xupper, curandState_t* random_state) 
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x; // TODO: is this correct?
    curandState_t localState = random_state[id];

    xj = xj + id * metropolis_offset + start_offset; // apply offset

    // double xjp = frand(xlower, xupper);
    double xjp = curand_uniform_double(&localState) * (xupper-xlower) + xlower;
    // printf("xjp: %f %i\n", xjp, (int)(xj[0]==xjp));
    // randomize_double_array(&xjp, 1, xlower, xupper, &localState);

    double S_delta = action_2p(xj[-1], xjp, xj[1]) - action_2p(xj[-1], *xj, xj[1]);

    if (S_delta < 0) {
        // if (fabs(*xj) < 0.15 && fabs(xj[-1]) < 0.15 && fabs(xj[1]) < 0.15) {
        // printf("%f, %f, %f | %f\n", xj[-1], *xj, xj[1], xjp);
        // printfl(S_delta);};
        // printf("test, %f, %f\n", *xj, xjp);
        *xj = xjp;
    }
    else {
        if (exp(-S_delta) > curand_uniform_double(&localState)) {
            // printf("a: %f %f\n", exp(-S_delta), test);
            *xj = xjp;
        };
    };

    random_state[id] = localState;
}


void metropolis_algo(
    double x0, double xN,
    size_t N_lattices, size_t N_measure, size_t N_montecarlo, size_t N_markov,
    const char filename[], const char equilibrium_filename[])
{
    size_t metropolis_offset = 2; // offset between kernels. The smaller the number, the more kernels run in parallel. Minimum 2
    size_t metropolis_kernels = (int)ceil( (double)(N-1) / metropolis_offset ); // amount of kernels that are run in parallel

    size_t N_measurements = N_lattices * N_measure;

    curandState_t* random_state;
    CUDA_CALL(cudaMallocManaged(&random_state, (N-1) * sizeof(curandState_t)));
    setup_randomize<<<1, N-1>>>(random_state);
    cudaDeviceSynchronize();
    
    double *x, *ensemble;
    CUDA_CALL(cudaMallocManaged(&x, (N+1) * sizeof(double)));
    // cudaMallocPitch(&ensemble, &ensemble_pitch, (N+1)*sizeof(double), N_measurements);
    CUDA_CALL(cudaMallocHost(&ensemble, (N+1) * N_measurements * sizeof(double)));
    size_t ensemble_pitch = (N+1)*sizeof(double);
    // double *ensemble = malloc(N_measurements * (N+1) * sizeof(double));

    // initialize boundary values
    x[0] = x0;
    x[N] = xN;

        // randomize_double_array<<<1, N-1>>>(x+1, N-1, xlower, xupper, random_state);
        // randomize_double_array<<<1, 128>>>(x+1, N-1, xlower, xupper, random_state);
        // test_fn<<<1, 123>>>(x+1, N-1);
        // cudaDeviceSynchronize();
        
    // metropolis algorithm
    unsigned int measure_index = 0;
    for (int l=0; l<N_lattices; l++) {
        randomize_double_array<<<1, N-1>>>(x+1, N-1, xlower, xupper, random_state);
        cudaDeviceSynchronize();

        for (size_t j=0; j<N_measure; j++) {
            for (size_t k=0; k<N_montecarlo; k++) {
                // for (size_t i=1; i<N; i++) {
                //     // N_markov metropolis steps on the lattice site 
                for (size_t start_offset=0; start_offset<metropolis_offset; start_offset++) {
                    // for (size_t o=0; o<N_markov; o++) {
                        metropolis_step<<<1, metropolis_kernels>>>(x+1, metropolis_offset, start_offset, xlower, xupper, random_state);
                        cudaDeviceSynchronize();
                    // };
                };
                // };
            };
            // measure the new lattice configuration
            CUDA_CALL(cudaMemcpy((float*)((char*)ensemble + ensemble_pitch*measure_index), x, (N+1)*sizeof(double), cudaMemcpyHostToHost));
            measure_index++;
        };
    };

    // write to csv
    if (filename) {
        FILE* file = fopen(filename, "w");
        export_csv_double_2d(file, ensemble, ensemble_pitch, N+1, N_measurements);
        fclose(file);
    }

    cudaFree(random_state);
    cudaFree(x);
    cudaFree(ensemble);
}



int main()
{
    srand(time(NULL));
    // srand(42);


    time_t time_start = time(NULL); // start measuring time

    // potential_ptr = *potential;

    //// Fig. 4, 5
    // m0 = 1.0;
    // mu_sq = 1.0;
    // lambda = 0.0;
    // N = 1000;
    // epsilon = 1.;
    // Delta = 2 * sqrt(epsilon);

    metropolis_algo(0., 0., 3, 5, 5, 5, "harmonic_a.csv", NULL);

/*
    //// Fig. 6
    m0 = 0.5;
    mu_sq = 2.0;
    lambda = 0.0;
    N = 51;
    epsilon = 0.5;
    Delta = 2 * sqrt(epsilon);
    metropolis_algo(0., 0., 1, 10, 1, 5, "harmonic_b.csv", NULL);
    // metropolis_algo(0., 0., 50, 1, 50, 5, "harmonic_b.csv", NULL);
    // metropolis_algo(0., 0., 6, 60, 5, 5, NULL, "harmonic_b.csv");

    // use the f_sq potential from here on
    potential_ptr = *potential_alt;

    //// Fig. 7
    m0 = 0.5;
    lambda = 1.0;
    epsilon = 1.0;
    N = 50;

    f_sq = 0.5;
    metropolis_algo(0., 0., 1, 1, 40, 5, "anharmonic_a.csv", NULL);
    f_sq = 1.0;
    metropolis_algo(0., 0., 1, 1, 40, 5, "anharmonic_b.csv", NULL);
    f_sq = 2.0;
    metropolis_algo(0., 0., 1, 1, 40, 5, "anharmonic_c.csv", NULL);

    //// Fig. 8
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

    time_t time_finish = time(NULL); // time measured until now


    const time_t total_time = difftime(time_finish, time_start);
    printf("total time taken: %fs\n", (double)total_time);




/*
    // bin the data
    double bin_lower = -5.;
    double bin_upper = 5.;
    const unsigned int N_bins = 30;

    double bins[N_bins];
    double bins_range[N_bins];

    bin_data(ensemble, N_lattices*(N+1), bins, N_bins, bin_lower, bin_upper);
    // bin_data(measurements, N_measurements*(N+1), bins, N_bins, bin_lower, bin_upper);
    bin_range(bins_range, N_bins, bin_lower, bin_upper);

    FILE* bin_file = fopen("bins.csv", "w");
    export_csv_double_1d(bin_file, N_bins, bins_range);
    export_csv_double_1d(bin_file, N_bins, bins);
    fclose(bin_file);

    printf("%i %i\n", N_measurements, measure_index);
    printf("%i %i\n", N_measurements, N+1);
*/
}