#include <metropolis.cuh>
#include <helper.cuh>

__device__ double potential(double x, metropolis_parameters params)
{
    if (params.alt_potential)
        return params.lambda * pow( (pow(x, 2.) - params.f_sq), 2. ); // alt potential
    else
        return 1./2. * params.mu_sq * pow(x, 2.) + params.lambda * pow(x, 4.);
}

__device__ double action_point(double x0, double x1, metropolis_parameters params)
{
    return params.a * (1./2. * params.m0 * pow((x1-x0), 2) / pow(params.a, 2) + potential(x0, params));
}

__device__ double action_2p(double xm1, double x0, double x1, metropolis_parameters parameters)
{
    double action_0 = action_point(xm1, x0, parameters);
    double action_m1 = action_point(x0, x1, parameters);
    return action_0 + action_m1;
}

__global__ void action_latticeconf_synchronous(double* lattice, metropolis_parameters params, double* action) 
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

__global__ void metropolis_step(double* xj, size_t start_offset, metropolis_parameters params, curandState_t* random_state) 
{ // TODO: try only making changes at the very end
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t offset = start_offset + idx * params.metropolis_offset;
    if (offset >= params.N) { // do nothing if the point would be out of range
        return;
    } 

    curandState_t localState = random_state[offset];

    double xjp = curand_uniform_double(&localState) * (2*params.Delta) - params.Delta + xj[offset];
    double lneighbor = xj[offset - 1];
    double rneighbor = xj[(offset + 1) % params.N];
    double S_delta = action_2p(lneighbor, xjp, rneighbor, params) - action_2p(lneighbor, xj[offset], rneighbor, params);

    if (S_delta <= 0) {
        xj[offset] = xjp;
    }
    else {
        double r = curand_uniform_double(&localState);
        if (exp(-S_delta) > r) {
            xj[offset] = xjp;
        };
    };

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
    
    printf("total: %zd\tblocks: %zd\t perblock: %zd\n", metropolis_kernels, metropolis_blocks, threads_per_block);

    // initialize data arrays
    size_t N_measurements = params.N_measure;

    curandState_t *random_state;
    CUDA_CALL(cudaMallocManaged(&random_state, N*sizeof(curandState_t)));
    setup_randomize<<<1, max_threads_per_block>>>(random_state, N, random_seed);
    cudaDeviceSynchronize();
    
    double *x, *ensemble;
    CUDA_CALL(cudaMallocManaged(&x, N*sizeof(double)));
    size_t ensemble_pitch;
    CUDA_CALL(cudaMallocPitch(&ensemble, &ensemble_pitch, N*sizeof(double), N_measurements));

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

    // cleanup and return
    CUDA_CALL(cudaFree(random_state));
    CUDA_CALL(cudaFree(x));
    *ensemble_out = ensemble;
    *pitch = ensemble_pitch;
    *width = N;
    *height = N_measurements;
}

void metropolis_allinone(metropolis_parameters params, const char* filename)
{
    double* ensemble;
    size_t pitch, width, height;
    metropolis_algo(params, &ensemble, &pitch, &width, &height);
    export_metropolis_data(filename, ensemble, pitch, width, height);
    CUDA_CALL(cudaFree(ensemble));
}