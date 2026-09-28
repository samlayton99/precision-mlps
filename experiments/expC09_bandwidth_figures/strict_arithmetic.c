/* Correctly rounded, p-bit centered-tanh inference. No fused operations. */
#include <mpfr.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

int strict_features(size_t n, size_t width, const double *x, const double *centers,
                    double gamma, unsigned p, double *out) {
    mpfr_t *c = malloc(width * sizeof(mpfr_t));
    if (!c) return 1;
    for (size_t j = 0; j < width; ++j) {
        mpfr_init2(c[j], p);
        mpfr_set_d(c[j], centers[j], MPFR_RNDN);
    }
    mpfr_t xx, g, delta, z, feature;
    mpfr_inits2(p, xx, g, delta, z, feature, (mpfr_ptr) 0);
    mpfr_set_d(g, gamma, MPFR_RNDN);
    /* Beyond this conservative threshold tanh rounds exactly to +/-1.
       1-tanh(z) < 2 exp(-2z) <= 2^(-p-2), less than half an ulp below 1. */
    double cutoff = (p + 3) * log(2.0) / 2;
    for (size_t i = 0; i < n; ++i) {
        mpfr_set_d(xx, x[i], MPFR_RNDN);
        for (size_t j = 0; j < width; ++j) {
            mpfr_sub(delta, xx, c[j], MPFR_RNDN);
            mpfr_mul(z, g, delta, MPFR_RNDN);
            double zd = mpfr_get_d(z, MPFR_RNDN);
            if (fabs(zd) >= cutoff) {
                out[i * width + j] = copysign(1.0, zd);
            } else {
                mpfr_tanh(feature, z, MPFR_RNDN);
                out[i * width + j] = mpfr_get_d(feature, MPFR_RNDN);
            }
        }
    }
    for (size_t j = 0; j < width; ++j) mpfr_clear(c[j]);
    free(c);
    mpfr_clears(xx, g, delta, z, feature, (mpfr_ptr) 0);
    return 0;
}

int strict_readout(size_t n, size_t width, const double *features,
                   const double *weights, double bias, unsigned p, double *out) {
    mpfr_t *w = malloc(width * sizeof(mpfr_t));
    if (!w) return 1;
    for (size_t j = 0; j < width; ++j) {
        mpfr_init2(w[j], p);
        mpfr_set_d(w[j], weights[j], MPFR_RNDN);
    }
    mpfr_t b, feature, product, sum;
    mpfr_inits2(p, b, feature, product, sum, (mpfr_ptr) 0);
    mpfr_set_d(b, bias, MPFR_RNDN);
    for (size_t i = 0; i < n; ++i) {
        mpfr_set(sum, b, MPFR_RNDN);
        for (size_t j = 0; j < width; ++j) {
            mpfr_set_d(feature, features[i * width + j], MPFR_RNDN);
            mpfr_mul(product, feature, w[j], MPFR_RNDN);
            mpfr_add(sum, sum, product, MPFR_RNDN);
        }
        out[i] = mpfr_get_d(sum, MPFR_RNDN);
    }
    for (size_t j = 0; j < width; ++j) mpfr_clear(w[j]);
    free(w);
    mpfr_clears(b, feature, product, sum, (mpfr_ptr) 0);
    return 0;
}
