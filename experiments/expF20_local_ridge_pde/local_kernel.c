/* Uniform-center tanh profiles. No feature matrix and no fast-math.
 * Values = exact step tails + truncated localized correction.
 * The transpose includes the exact transpose of the prefix-sum tails. */
#include <math.h>
#include <stdlib.h>

static int clamp(int a, int n) { return a < 0 ? 0 : (a > n ? n : a); }
static double feature(double e, int positive, int order, double gamma) {
    double denom = 1.0 + e;
    if (order == 0) return positive ? -2.0/denom : 2.0*e/denom;
    double d = 4.0*e/(denom*denom);
    if (order == 1) return gamma*d;
    return -2.0*gamma*gamma*((e-1.0)/denom)*d;
}

void profiles(int m, int n, int k, const double *projection,
              const double *weights, double c0, double h, double gamma,
              double cutoff, int order, double *output) {
    double *prefix = malloc((n+1)*sizeof(double));
    double ratio = exp(-2.0*gamma*h);
    for (int a=0; a<m; ++a) {
        const double *w = weights + (size_t)a*n;
        if (order == 0) {
            prefix[0]=0.;
            for (int j=0; j<n; ++j) prefix[j+1]=prefix[j]+w[j];
        }
        for (int b=0; b<k; ++b) {
            double t=projection[(size_t)a*k+b];
            int split=clamp((int)floor((t-c0)/h)+1,n);
            int lo=clamp((int)ceil((t-cutoff/gamma-c0)/h),n);
            int hi=clamp((int)floor((t+cutoff/gamma-c0)/h)+1,n);
            double sum=order == 0 ? 2.0*prefix[split]-prefix[n] : 0.;
            double e=exp(2.0*gamma*(t-(c0+lo*h)));
            for (int j=lo; j<hi; ++j) {
                sum+=w[j]*feature(e,j<split,order,gamma);
                e*=ratio;
            }
            output[(size_t)a*k+b]=sum;
        }
    }
    free(prefix);
}

void transpose(int m, int n, int k, const double *projection,
               const double *residual, double c0, double h, double gamma,
               double cutoff, int order, double *output) {
    double ratio=exp(-2.0*gamma*h);
    double *events=calloc(n+1,sizeof(double));
    for (int a=0; a<m; ++a) {
        double *w=output+(size_t)a*n;
        for (int j=0; j<n; ++j) w[j]=0.;
        for (int j=0; j<=n; ++j) events[j]=0.;
        for (int b=0; b<k; ++b) {
            double t=projection[(size_t)a*k+b];
            double z=residual[(size_t)a*k+b];
            int split=clamp((int)floor((t-c0)/h)+1,n);
            int lo=clamp((int)ceil((t-cutoff/gamma-c0)/h),n);
            int hi=clamp((int)floor((t+cutoff/gamma-c0)/h)+1,n);
            if (order == 0) { events[0]+=z; events[split]-=2.0*z; }
            double e=exp(2.0*gamma*(t-(c0+lo*h)));
            for (int j=lo; j<hi; ++j) {
                w[j]+=z*feature(e,j<split,order,gamma);
                e*=ratio;
            }
        }
        if (order == 0) {
            double sum=0.;
            for (int j=0; j<n; ++j) { sum+=events[j]; w[j]+=sum; }
        }
    }
    free(events);
}
