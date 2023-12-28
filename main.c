#include <stdlib.h>
#include <stdio.h>
// #include <math.h>
#include <complex.h>

// #ifdef _Imaginary_I
// #define __STDC_IEC_559_COMPLEX__
// #endif

// global variables
const int N = 1e2;
const double m0;
const double epsilon;
double complex a;
const double mu;
const double lambda;


// helper functions
double frand(double lower, double upper)
{
    return lower + (upper - lower) * ((double)rand() / RAND_MAX);
}

// big functions
double complex action(double x0, double x1)
{
    // double V = 1/2 * pow(2, mu) * pow(2, x0) + lambda * pow(4, x0); // anharmonic oscillator potential
    double V = 0.;
    return a * (m0 * (x1-x0) / a + V);
}


// general
// N = 1e2;
m0 = 1.0;
// time step
epsilon = 0.1;
a = I * epsilon;
// potential
mu = 1.0;
lambda = 0.;


int main()
{
    double x[N];

    for (int i=0; i<N; i++) {
        x[i] = frand(0., 1.);
        // printf("%i %f \n", i, x[i]);
        printf("%c\n", action(x[i], x[i+1]));
    };

    // double imaginary a = I * epsilon;
}