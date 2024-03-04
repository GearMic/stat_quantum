#include <helper.cuh>
#include <metropolis.cuh>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include <curand_kernel.h>

 
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
        printf("\n");
    }


    metropolis_parameters params; 
    params.kernel_offset = 2;
    params.xlower = -5;
    params.xupper = 5;
    params.x0 = 0.0;
    params.a = 1.;
    params.N = 1000;
    params.N_until_equilibrium = 100;
    params.N_measure = 60;
    params.N_montecarlo = 10;
    params.N_markov = 1;
    params.Delta = 2.0;
    params.m0 = 1.0;
    params.lambda = 0.0;
    params.mu_sq = 1.0;
    params.f_sq = -1.0; // placeholder value
    params.alt_potential = false;

    // step 1: plot action
    metropolis_parameters params_0 = params;
    params_0.m0 = .5;
    params_0.a = .5;
    params_0.N = 1000; // broken for N>1026?
    params_0.N_until_equilibrium = 0; 
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

    size_t n_blocks = ceil_division(params_0.N-1, max_threads_per_block);

    for (size_t i=0; i<height; i++) {
        action_latticeconf<<<1, 1>>>((double*)((char*)ensemble + i*pitch), params_0, actions+i);
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
    params_6.N_montecarlo = 50;
    params_6.N_markov = 1;
    params_6.mu_sq = 2.0;
    params_6.lambda = 0.0;
    params_6.a = 0.5;
    params_6.Delta = 2 * sqrt(params_6.a);
    metropolis_allinone(params_6, "harmonic_b.csv");

    // Fig. 7
    metropolis_parameters params_7 = params;
    params_7.alt_potential = true;
    params_7.N = 50;
    params_7.N_measure = 1;
    params_7.N_montecarlo = 40;
    params_7.N_markov = 5;
    params_7.lambda = 1.0;
    params_7.a = 1.0;
    params_7.Delta = 2 * sqrt(params.a);
    params_7.m0 = 0.5;

    params_7.f_sq = 0.5;
    metropolis_allinone(params_7, "anharmonic_a.csv");
    params_7.f_sq = 1.0;
    metropolis_allinone(params_7, "anharmonic_b.csv");
    params_7.f_sq = 2.0;
    metropolis_allinone(params_7, "anharmonic_c.csv");

    // Fig. 8
    metropolis_parameters params_8 = params;
    params_8.alt_potential = true;
    params_8.xlower = -5.;
    params_8.xupper = 5.;
    params_8.N_until_equilibrium = 10000;
    params_8.m0 = 0.5;
    params_8.lambda = 1.0;
    params_8.f_sq = 2.0;
    params_8.N = 200;
    params_8.a = 0.25;
    params_8.Delta = 2. * sqrt(params_8.a);
    params_8.N_measure = 5000;
    params_8.N_montecarlo = 10;
    metropolis_allinone(params_8, "anharmonic_d.csv");
    // metropolis_algo(0., 0., 100, 50, 10, 5, NULL, "anharmonic_d.csv");
    // metropolis_algo(0., 0., 100, 50, 1, 5, "anharmonic_d.csv", NULL);

    // Fig. 9
    metropolis_parameters params_9 = params;
    params_9.alt_potential = true;
    params_9.N_until_equilibrium = 200;
    params_9.N_measure = 500;
    params_9.lambda = 1.0;
    params_9.m0 = 0.5;
    params_9.f_sq = 2.0;
    params_9.N = 303;
    params_9.a = 0.25;
    params_9.Delta = 2. * sqrt(params_9.a);
    params_9.N_montecarlo = 5;
    metropolis_allinone(params_9, "anharmonic_correlation_a.csv");
    params_9.N_montecarlo = 10;
    metropolis_allinone(params_9, "anharmonic_correlation_b.csv");
    params_9.N_montecarlo = 15;
    metropolis_allinone(params_9, "anharmonic_correlation_c.csv");
    params_9.N_montecarlo = 1;
    metropolis_allinone(params_9, "anharmonic_correlation_d.csv");

    // Fig. 10
    metropolis_parameters params_10 = params_9;
    params_10.N_until_equilibrium = 200;
    params_10.N_measure = 500;
    params_10.N_montecarlo = 1;
    params_10.a = 0.25;
    params_10.Delta = 2. * sqrt(params_9.a);
    for (size_t i=0; i<5; i++) {
        params_10.f_sq = (double)i * 0.5;
        char filename[] = "anharmonic_energy_";
        sprintf(filename, "anharmonic_energy%zd.csv", i);
        metropolis_allinone(params_10, filename);
    }
} 
