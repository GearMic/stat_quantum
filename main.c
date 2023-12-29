#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>

// #ifdef _Imaginary_I
// #define __STDC_IEC_559_COMPLEX__
// #endif

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
Nt = 20 // number of Monte Carlo iterations
Ne = 1 // number of initial lattice configurations generated TODO: is this correct?
Ni = 3 // Number (interval) of Markov iterations between measurements TODO: is this correct?




// big functions
double complex action(double x0, double x1)
{
    double V = 1./2. * pow(2, mu) * pow(2, x0) + lambda * pow(4, x0); // anharmonic oscillator potential
    printf("V: %f \n", V);
    return a * (m0 * (x1-x0) / a + V);
}

double metropolis_step(double* xj) 
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


// helper functions
double frand(double lower, double upper)
{
    return lower + (upper - lower) * ((double)rand() / RAND_MAX);
}

void printc(double complex z)
{
    printf("%f + i%f\n", creal(z), cimag(z));
}


// // general
// N = 1e2;
// m0 = 1.0;
// // time step
// epsilon = 0.1;
// a = I * epsilon;
// // potential
// mu = 1.0;
// lambda = 0.;


int main()
{
    double x[N];

    for (int i=0; i<N; i++) {
        x[i] = frand(xlower, xupper);
        // printf("%i %f \n", i, x[i]);
        printc(action(x[i], x[i+1]));
    };

    // double imaginary a = I * epsilon;
}