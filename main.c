// clang -o a main.c -lm

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <time.h>

//// parameters
int N;
double epsilon;
double a;
double Delta;
// double complex a = I * epsilon;
double m0;
// potential
double mu_sq;
double lambda;
double f_sq;
// x range
const double xlower = -2.;
const double xupper = 2.;

double (*potential_ptr)(double);



// // simulation parameters
// const unsigned int N_measure = 60; // number of measurements made after monte carlo iterations. influences the total amount of M.C. iterations made
// // const unsigned int Nt = 20; // number of Monte Carlo iterations
// const unsigned int N_lattices = 1; // number of initial lattice configurations generated
// const unsigned int N_montecarlo = 5; // see below (4.8) // Number of Monte Carlo iterations between measurements
// const unsigned int N_markov = 5; // n-bar from (3.29) // Number of Markov iterations on each lattice point
// double Delta; // initialized in main

// // boundary values
// const double x0 = 0.0;
// const double xN = 0.0;
// general eral
// const int N = 1e2;
// const double epsilon = .1;


// helper functions
double frand(double lower, double upper)
{
    // static int seed;
    // seed = rand();
    // srand(seed);
    return lower + (upper - lower) * ((double)rand() / (double)RAND_MAX);
}

void randomize_double_array(double* array, unsigned int len, double lower, double upper)
{
    for (unsigned int i=0; i<len; i++) {
        array[i] = frand(xlower, xupper);
        // array[i] = 0.; // for testing
    };
}

void printfl(double x)
{
    printf("%f\n", x);
}

void printc(double complex z)
{
    printf("%f + i%f\n", creal(z), cimag(z));
}

void export_csv_double_1d(FILE* file, const unsigned int cols, double arr[cols])
{
    for (int col=0; col<cols; col++) {
        fprintf(file, "%f%s", arr[col], (col==cols-1 ? "":","));
    };
    fprintf(file, "\n");
}

void export_csv_double_2d(FILE* file, const unsigned int rows, const unsigned int cols, double arr[rows][cols])
{
    for (int row=0; row<rows; row++) {
        export_csv_double_1d(file, cols, arr[row]);
    };
}

void bin_data(double x[], unsigned int N_x, double bins[], unsigned int N_bins, double xlower, double xupper)
// formula 4.15 // TODO: is this correct?
{
    // initialize bins
    for (int j=0; j<N_bins; j++) {
        bins[j] = 0.;
    }

    double bin_size = (xupper-xlower) / (double)N_bins;
    // fill bins
    for (int i=0; i<N_x; i++) {
        double xi = x[i];
        for (int j=0; j<N_bins; j++) {
            if (xlower + j * bin_size <= xi && xlower + (j+1) * bin_size > xi) {
                bins[j] += 1. / bin_size / (double)N_x; // TODO: can this be done more efficiently?
                break;
            }
        }
    }
}

void bin_range(double range[], unsigned int N_bins, double xlower, double xupper)
// fill array of corresponding x values for data bins
{
    double bin_size = (xupper-xlower) / (double)N_bins;
    for (int j=0; j<N_bins; j++) {
        range[j] = xlower + j * bin_size;
    }
}



// big functions
double potential(double x)
{
    return 1./2. * pow(mu_sq, 2) * pow(x, 2) + lambda * pow(x, 4); // anharmonic oscillator potential
}

double potential_alt(double x)
{
    return lambda * pow( pow(x, 2.f) - f_sq, 2.f );
}

double action_point(double x0, double x1)
{
    return epsilon * (1./2. * m0 * pow((x1-x0), 2) / pow(epsilon, 2) + (*potential_ptr)(x0));
}

double action(double* x, unsigned int N)
{
    double action = 0.;
    for (int i=0; i<=N; i++) {
        action += action_point(x[i-1], x[i]);
    }
    return action;
}

double action_2p(double xm1, double x0, double x1)
{
    double action_0 = action_point(xm1, x0);
    double action_m1 = action_point(x0, x1);
    return action_0 + action_m1;
}

double complex c_action_point(double x0, double x1)
{
    return a * (1./2. * m0 * cpow((x1-x0), 2) / cpow(a, 2) + (*potential_ptr)(x0));
}

double complex c_action_2p(double xm1, double x0, double x1)
{
    double complex action_0 = c_action_point(xm1, x0);
    double complex action_m1 = c_action_point(x0, x1);
    return action_0 + action_m1;
}

void metropolis_step(double* xj) 
{
    // double xjp = frand(xlower, xupper);
    double xjp = frand(*xj - Delta, *xj + Delta); // xj-prime

    double S_delta = action_2p(xj[-1], xjp, xj[1]) - action_2p(xj[-1], *xj, xj[1]);

    if (S_delta < 0) {
        // if (fabs(*xj) < 0.15 && fabs(xj[-1]) < 0.15 && fabs(xj[1]) < 0.15) {
        // printf("%f, %f, %f | %f\n", xj[-1], *xj, xj[1], xjp);
        // printfl(S_delta);};
        *xj = xjp;
    }
    else {
        double test = frand(0., 1.);
        // if (exp(-S_delta) > frand(0., 1.)) {
        if (exp(-S_delta) > test) {
            // printf("a: %f %f\n", exp(-S_delta), test);
            *xj = xjp;
        };
    };
}

void metropolis_algo(
    double x0, double xN,
    unsigned int N_lattices, unsigned int N_measure, unsigned int N_montecarlo, unsigned int N_markov,
    char filename[], char equilibrium_filename[])
    // double ensemble[N_lattices*(1+N_measure)][N+1], double equilibrium_ensemble[N_lattices][N+1])
{
    // ensemble
    // const unsigned int N_measurements = N_lattices * (1 + N_measure); // NOTE: for including the initial random lattice configurations
    const unsigned int N_measurements = N_lattices * N_measure;
    double ensemble[N_measurements][N+1];
    double equilibrium_ensemble[N_lattices][N+1]; // array containing the "finished" states;


    double x[N+1];
    // initialize boundary values
    x[0] = x0;
    x[N] = xN;

    // // measure initial lattice configuration
    // randomize_double_array(x+1, N-1, xlower, xupper);
    // memcpy(ensemble[0], x, (N+1)*sizeof(double));

    // metropolis algorithm
    unsigned int measure_index = 0;
    for (int l=0; l<N_lattices; l++) {
        randomize_double_array(x+1, N-1, xlower, xupper);
        // // measure initial lattice configuration
        // memcpy(ensemble[0], x, (N+1)*sizeof(double));
        // measure_index++;

        for (int j=0; j<N_measure; j++) {
            for (int k=0; k<N_montecarlo; k++) {
                for (int i=1; i<N; i++) {
                    // N_markov metropolis steps on the lattice site 
                    for (int o=0; o<N_markov; o++) {
                        metropolis_step(x+i);
                    }
                };
            };
            // measure the new lattice configuration
            memcpy(ensemble[measure_index], x, (N+1)*sizeof(double));
            measure_index++;
        };
        memcpy(equilibrium_ensemble[l], x, (N+1)*sizeof(double));
    }

    // write to csv
    if (filename) {
        FILE* file = fopen(filename, "w");
        export_csv_double_2d(file, N_measurements, N+1, ensemble);
        fclose(file);
    }

    if (equilibrium_filename) {
        FILE* equilibrium_file = fopen(equilibrium_filename, "w");
        export_csv_double_2d(equilibrium_file, N_measurements, N+1, ensemble);
        fclose(equilibrium_file);
    }
}

// double correlation_function(unsigned int rows, unsigned int cols, double ensemble[rows][cols], )



int main()
{
    srand(time(NULL));
    // srand(42);

    potential_ptr = *potential;

    //// Fig. 4, 5
    m0 = 1.0;
    mu_sq = 1.0;
    lambda = 0.0;
    N = 1000;
    epsilon = 1.;
    Delta = 2 * sqrt(epsilon);
    metropolis_algo(0., 0., 3, 60, 5, 5, "harmonic_a.csv", NULL);

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