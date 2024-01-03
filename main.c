// clang -o a main.c -lm

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <string.h>

//// paramters
// boundary
const double x0 = 0.0;
const double xN = 0.0;
// general
const int N = 1e2;
const double m0 = 1.0;
// time step
const double epsilon = 0.1;
double complex a = I * epsilon;
// potential
const double mu = 1.0;
const double lambda = 0.;
// x range
const double xlower = -2.; // is this needed?
const double xupper = 2.;
// simulation parameters
const unsigned int Nt = 20; // number of Monte Carlo iterations
const unsigned int Ne = 3; // number of initial lattice configurations generated TODO: is this correct?
const unsigned int N_montecarlo = 10; // Number of Monte Carlo iterations between measurements
const unsigned int N_markov = 1; // Number of Markov iterations on each lattice point



// helper functions
double frand(double lower, double upper)
{
    return lower + (upper - lower) * ((double)rand() / RAND_MAX);
}

void printc(double complex z)
{
    printf("%f + i%f\n", creal(z), cimag(z));
}

void export_csv_double_2d(FILE* file, double arr[][N+1], const unsigned int rows, const unsigned int cols) // TODO: find a better way to pass the array
{
    for (int row=0; row<rows; row++) {
        for (int col=0; col<cols; col++) {
            fprintf(file, "%f%s", arr[row][col], (col==cols-1 ? "":","));
            // fprintf(file, "%f%s", (double)1, (col==cols-1 ? "":","));
            // fprintf(file, "l, ");
        };
        fprintf(file, "\n");
    };
    // fprintf(file, "test");
}



// big functions
// double complex action(double x0, double x1)
// {
//     double V = 1./2. * pow(2, mu) * pow(2, x0) + lambda * pow(4, x0); // anharmonic oscillator potential
//     return a * (m0 * (x1-x0) / cpow(a, 2) + V);
// }

double action(double x0, double x1)
{
    double V = 1./2. * pow(2, mu) * pow(2, x0) + lambda * pow(4, x0); // anharmonic oscillator potential
    return epsilon * (m0 * (x1-x0) / cpow(epsilon, 2) + V);
}

void metropolis_step(double* xj) 
{
    double xjp = frand(xlower, xupper);
    // double S_delta = cabs(action(*xj, xjp)) - cabs(action(*xj, *(xj+1)));
    // double S_delta = cabs(action(*xj, xjp) - action(*xj, *(xj+1)));
    double S_delta = action(*xj, xjp) - action(*xj, *(xj+1));

    if (S_delta <= 0) {
        *xj = xjp;
    }
    else {
        if (exp(-S_delta) > frand(0., 1.)) {
            *xj = xjp;
        };
    };
}



int main()
{
    printf("a\n");

    double x[N+1];
    // const unsigned int N_measurements = Ne * (1 + Nt * (N-1));
    // const unsigned int N_measurements = 1 + Ne * Nm;
    const unsigned int N_measurements = Ne * (1 + Nt);
    double measurements[N_measurements][N+1];

    // double x[N+1];
    // const unsigned int N_measurements = Ne * (1 + Nt * (N-1));
    // double measurements[N_measurements][N+1];

    printf("a\n");

    for (int i=0; i<N+1; i++) {
        x[i] = frand(xlower, xupper);
        // printf("%i %f \n", i, x[i]);
        // printc(action(x[i], x[i+1]));
    };

    printf("a\n");
    
    // initialize boundary values
    x[0] = x0;
    x[N] = xN;
    for (int i=0; i<N_measurements; i++) {
        measurements[i][0] = x0;
        measurements[i][N] = xN;
    }

    printf("a\n");

    // metropolis algorithm
    memcpy(measurements[0], x, (N+1)*sizeof(double)); // measure initial lattice configuration
    unsigned int measure_index = 1;
    for (int l=0; l<Ne; l++) {
        for (int j=0; j<Nt; j++) {
            for (int k=0; k<N_montecarlo; k++) {
            for (int i=1; i<N; i++) {
                // Ni metropolis steps on the lattice site 
                for (int o=0; o<N_markov; o++) {
                    metropolis_step(x+i);
                }
            };
            };
            // measure the new lattice configuration
            memcpy(measurements[measure_index], x, (N+1)*sizeof(double));
            measure_index++;
        };
    }
    printf("%i %i\n", N_measurements, measure_index);
    printf("%i %i\n", N_measurements, N+1);
    
    printf("a\n");

    // write to csv
    FILE* file = fopen("out.csv", "w");
    export_csv_double_2d(file, measurements, N_measurements, N+1);
    fclose(file);

    printf("a\n");
}