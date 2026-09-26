/* Native float32 centered-tanh inference, with separate multiply/add. */
#include <math.h>
#include <stddef.h>

void native_fp32_forward(size_t n, size_t width, const float *x,
                         const float *centers, float gamma, const float *weights,
                         float bias, float *out) {
    for (size_t i = 0; i < n; ++i) {
        float sum = bias;
        for (size_t j = 0; j < width; ++j) {
            float delta = x[i] - centers[j];
            float argument = gamma * delta;
            float feature = tanhf(argument);
            float product = weights[j] * feature;
            sum = sum + product;
        }
        out[i] = sum;
    }
}
