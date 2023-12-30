// clang -o a main.c -lm

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <string.h>

//// paramters
// boundary
const double x0 = 5.0;
const double xN = 5.0;
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
const double xlower = 0.; // is this needed?
const double xupper = 1.;
// simulation parameters
const unsigned int Nt = 20; // number of Monte Carlo iterations
const unsigned int Ne = 1; // TODO: unused right now // number of initial lattice configurations generated TODO: is this correct?
const unsigned int Ni = 3; // Number (interval) of Markov iterations between measurements TODO: is this correct?



// helper functions
double frand(double lower, double upper)
{
    return lower + (upper - lower) * ((double)rand() / RAND_MAX);
}

void printc(double complex z)
{
    printf("%f + i%f\n", creal(z), cimag(z));
}



// big functions
double complex action(double x0, double x1)
{
    double V = 1./2. * pow(2, mu) * pow(2, x0) + lambda * pow(4, x0); // anharmonic oscillator potential
    return a * (m0 * (x1-x0) / a + V);
}

void metropolis_step(double* xj) 
{
    double xjp = frand(xlower, xupper);
    // double S_xj = action(*xj, *(xj+1));
    // double S_xjp = action(*xj, xjp);
    // double S_delta = S_xjp - S_xj;
    double S_delta = cabs(action(*xj, xjp)) - cabs(action(*xj, *(xj+1))); // TODO: is this correct?

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
    double x[N+1];
    const unsigned int N_measurements = 1+Nt*(N-1);
    double measurements[N_measurements][N+1];

    for (int i=0; i<N+1; i++) {
        x[i] = frand(xlower, xupper);
        // printf("%i %f \n", i, x[i]);
        // printc(action(x[i], x[i+1]));
    };

    // initialize boundary values
    x[0] = x0;
    x[N] = xN;
    for (int i=0; i<N_measurements; i++) {
        measurements[i][0] = x0;
        measurements[i][N] = xN;
    }

    // metropolis algorithm
    memcpy(measurements[0], x, (N+1)*sizeof(double)); // measure initial lattice configuration
    for (int j=0; j<Nt; j++) {
        for (int i=1; i<N; i++) {
            // Ni metropolis steps on the lattice site 
            for (int k=0; k<Ni; k++) {
                metropolis_step(x+i);
            }
            // measure the new lattice configuration
            memcpy(measurements[1+j*(N-1)+i-1], x, (N+1)*sizeof(double)); // TODO: fix the index
        };
    };
}