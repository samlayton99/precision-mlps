/* All numerical temporaries have precision p; no fused multiply-add. */
#include <mpfr.h>
#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#define RN MPFR_RNDN

static mpfr_t *vector(size_t n, unsigned p) {
    mpfr_t *v = malloc(n*sizeof(mpfr_t));
    if (!v) return NULL;
    for (size_t i=0; i<n; ++i) { mpfr_init2(v[i],p); mpfr_set_zero(v[i],0); }
    return v;
}
static void release(mpfr_t *v, size_t n) {
    if (!v) return;
    for (size_t i=0; i<n; ++i) mpfr_clear(v[i]);
    free(v);
}

int affine_features(size_t n, size_t w, const double *x, const double *weight,
                    const double *bias, unsigned p, double *out) {
    mpfr_t *a=vector(w,p), *b=vector(w,p);
    if (!a || !b) { release(a,w); release(b,w); return 1; }
    for (size_t j=0;j<w;++j) { mpfr_set_d(a[j],weight[j],RN); mpfr_set_d(b[j],bias[j],RN); }
    mpfr_t xx,z,t;
    mpfr_inits2(p,xx,z,t,(mpfr_ptr)0);
    /* Conservative exact saturation shortcut; cutoff is control logic only. */
    const double cutoff=(p+3)*log(2.)/2;
    for (size_t i=0;i<n;++i) {
        mpfr_set_d(xx,x[i],RN);
        for (size_t j=0;j<w;++j) {
            mpfr_mul(z,xx,a[j],RN);
            mpfr_add(z,z,b[j],RN);
            double zd=mpfr_get_d(z,RN);
            if (fabs(zd)>=cutoff) out[i*w+j]=copysign(1.,zd);
            else { mpfr_tanh(t,z,RN); out[i*w+j]=mpfr_get_d(t,RN); }
        }
    }
    release(a,w); release(b,w);
    mpfr_clears(xx,z,t,(mpfr_ptr)0);
    return 0;
}

int polynomial_data(size_t n, unsigned degree, const double *nodes,
                    const double *labels, double bias, unsigned p,
                    double *cheb_out, double *poly_out, double *taylor_out) {
    size_t size=degree+1;
    mpfr_t *coeff=vector(size,p), *prev=vector(size,p), *cur=vector(size,p),
           *next=vector(size,p), *poly=vector(size,p), *taylor=vector(size,p);
    if (!coeff || !prev || !cur || !next || !poly || !taylor) {
        release(coeff,size); release(prev,size); release(cur,size); release(next,size);
        release(poly,size); release(taylor,size); return 1;
    }
    mpfr_t x,y,t0,t1,t2,tmp,product,total,denom,two,one,bb;
    mpfr_inits2(p,x,y,t0,t1,t2,tmp,product,total,denom,two,one,bb,(mpfr_ptr)0);
    mpfr_set_ui(two,2,RN); mpfr_set_ui(one,1,RN);
    /* Direct discrete Chebyshev projection, with p-bit recurrence and sums. */
    for (size_t i=0;i<n;++i) {
        mpfr_set_d(x,nodes[i],RN); mpfr_set_d(y,labels[i],RN);
        mpfr_set(t0,one,RN); mpfr_set(t1,x,RN);
        mpfr_add(coeff[0],coeff[0],y,RN);
        if (degree) { mpfr_mul(product,y,t1,RN); mpfr_add(coeff[1],coeff[1],product,RN); }
        for (unsigned k=2;k<=degree;++k) {
            mpfr_mul(tmp,two,x,RN); mpfr_mul(tmp,tmp,t1,RN); mpfr_sub(t2,tmp,t0,RN);
            mpfr_mul(product,y,t2,RN); mpfr_add(coeff[k],coeff[k],product,RN);
            mpfr_set(t0,t1,RN); mpfr_set(t1,t2,RN);
        }
    }
    mpfr_set_ui(denom,n,RN);
    for (unsigned k=0;k<=degree;++k) {
        if (k) mpfr_mul(coeff[k],coeff[k],two,RN);
        mpfr_div(coeff[k],coeff[k],denom,RN);
        cheb_out[k]=mpfr_get_d(coeff[k],RN);
    }
    /* Chebyshev-to-monomial recurrence, saving every partial truncation. */
    mpfr_set_ui(cur[0],1,RN);
    for (unsigned k=0;k<=degree;++k) {
        for (unsigned j=0;j<=degree;++j) {
            mpfr_mul(product,coeff[k],cur[j],RN);
            mpfr_add(poly[j],poly[j],product,RN);
            poly_out[k*size+j]=mpfr_get_d(poly[j],RN);
        }
        if (k==0) {
            for (unsigned j=0;j<=degree;++j) mpfr_set(prev[j],cur[j],RN);
            mpfr_set_zero(cur[0],0);
            if (degree) mpfr_set_ui(cur[1],1,RN);
        } else {
            for (unsigned j=0;j<=degree;++j) {
                if (j) mpfr_mul(tmp,two,cur[j-1],RN); else mpfr_set_zero(tmp,0);
                mpfr_sub(next[j],tmp,prev[j],RN);
            }
            for (unsigned j=0;j<=degree;++j) { mpfr_set(prev[j],cur[j],RN); mpfr_set(cur[j],next[j],RN); }
        }
    }
    /* c_r=tanh^(r)(b)/r!: (r+1)c_(r+1)=delta_(r,0)-sum c_j*c_(r-j). */
    mpfr_set_d(bb,bias,RN); mpfr_tanh(taylor[0],bb,RN);
    for (unsigned r=0;r<degree;++r) {
        mpfr_set_zero(total,0);
        for (unsigned j=0;j<=r;++j) {
            mpfr_mul(product,taylor[j],taylor[r-j],RN); mpfr_add(total,total,product,RN);
        }
        if (r==0) mpfr_sub(tmp,one,total,RN); else mpfr_neg(tmp,total,RN);
        mpfr_set_ui(denom,r+1,RN); mpfr_div(taylor[r+1],tmp,denom,RN);
    }
    for (unsigned r=0;r<=degree;++r) taylor_out[r]=mpfr_get_d(taylor[r],RN);
    release(coeff,size); release(prev,size); release(cur,size); release(next,size);
    release(poly,size); release(taylor,size);
    mpfr_clears(x,y,t0,t1,t2,tmp,product,total,denom,two,one,bb,(mpfr_ptr)0);
    return 0;
}

int stencil_network(unsigned degree, const double *poly, const double *taylor,
                    double step, unsigned p, double *slopes, double *weights) {
    size_t width=2*degree+1;
    mpfr_t *a=vector(width,p);
    if (!a) return 1;
    mpfr_t h,half,v,c,term,denom,index,ratio,two;
    mpfr_inits2(p,h,half,v,c,term,denom,index,ratio,two,(mpfr_ptr)0);
    mpfr_set_d(h,step,RN); mpfr_set_ui(two,2,RN); mpfr_div(half,h,two,RN);
    int status=0;
    for (unsigned r=0;r<=degree;++r) {
        if (poly[r]==0.) continue;
        if (!isfinite(poly[r]) || !isfinite(taylor[r]) || taylor[r]==0.) {status=2;break;}
        mpfr_set_d(v,poly[r],RN); mpfr_set_d(c,taylor[r],RN); mpfr_div(term,v,c,RN);
        for (unsigned k=1;k<=r;++k) {
            mpfr_set_ui(index,k,RN); mpfr_mul(denom,h,index,RN); mpfr_div(term,term,denom,RN);
        }
        if (r%2) mpfr_neg(term,term,RN);
        for (unsigned j=0;j<=r;++j) {
            size_t slot=degree+2*j-r;
            mpfr_add(a[slot],a[slot],term,RN);
            if (j<r) {
                mpfr_set_ui(v,r-j,RN); mpfr_set_ui(denom,j+1,RN);
                mpfr_div(ratio,v,denom,RN); mpfr_mul(term,term,ratio,RN); mpfr_neg(term,term,RN);
            }
        }
    }
    if (!status) for (size_t j=0;j<width;++j) {
        mpfr_set_si(index,(long)j-(long)degree,RN); mpfr_mul(v,half,index,RN);
        slopes[j]=mpfr_get_d(v,RN); weights[j]=mpfr_get_d(a[j],RN);
        /* FP64 arrays are containers only: reject values they cannot store exactly. */
        if (!isfinite(weights[j]) || mpfr_cmp_d(a[j],weights[j]) || mpfr_cmp_d(v,slopes[j])) {status=3;break;}
    }
    release(a,width);
    mpfr_clears(h,half,v,c,term,denom,index,ratio,two,(mpfr_ptr)0);
    return status;
}
