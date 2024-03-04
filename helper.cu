#include <helper.cuh>

__global__ void setup_randomize(curandState_t* state, size_t len, unsigned long long seed)
{
    size_t id = blockDim.x * blockIdx.x + threadIdx.x; // TODO: is this correct?
    size_t stride = blockDim.x;

    for (unsigned int i=id; i<len; i+=stride) {
        curand_init(seed, i, 0, &state[i]);
    };
}

__global__ void randomize_double_array(double* array, size_t len, double lower, double upper, curandState_t* state)
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

size_t ceil_division(size_t dividend, size_t divisor)
// return lowest integer that is larger or equal to dividend/divisor
{
    return (int)ceil( (double)(dividend) / divisor );
}