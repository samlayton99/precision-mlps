/* Numerical rescue variants. Every floating-point primitive remains p-bit.
   Baseline kernels are included unchanged, preserving their experiment. */
#include "pbit_kernels.c"

/* Knuth TwoSum and Dekker/Veltkamp TwoProduct; no FMA or wider temporary. */
static void twosum(mpfr_t s, mpfr_t e, const mpfr_t a, const mpfr_t b, mpfr_t *t) {
    mpfr_add(s,a,b,RN);
    mpfr_sub(t[0],s,a,RN);
    mpfr_sub(t[1],s,t[0],RN);
    mpfr_sub(t[1],a,t[1],RN);
    mpfr_sub(t[2],b,t[0],RN);
    mpfr_add(e,t[1],t[2],RN);
}
static void twoproduct(mpfr_t prod, mpfr_t err, const mpfr_t a, const mpfr_t b,
                       const mpfr_t split, mpfr_t *t) {
    mpfr_mul(prod,a,b,RN);
    mpfr_mul(t[0],split,a,RN); mpfr_sub(t[1],t[0],a,RN);
    mpfr_sub(t[2],t[0],t[1],RN); mpfr_sub(t[3],a,t[2],RN);
    mpfr_mul(t[0],split,b,RN); mpfr_sub(t[1],t[0],b,RN);
    mpfr_sub(t[4],t[0],t[1],RN); mpfr_sub(t[5],b,t[4],RN);
    mpfr_mul(t[0],t[2],t[4],RN); mpfr_sub(t[1],prod,t[0],RN);
    mpfr_mul(t[0],t[3],t[4],RN); mpfr_sub(t[1],t[1],t[0],RN);
    mpfr_mul(t[0],t[2],t[5],RN); mpfr_sub(t[1],t[1],t[0],RN);
    mpfr_mul(t[0],t[3],t[5],RN); mpfr_sub(err,t[0],t[1],RN);
}
static void neumaier(mpfr_t sum,mpfr_t correction,const mpfr_t value,mpfr_t *t) {
    mpfr_add(t[0],sum,value,RN);
    if (mpfr_cmpabs(sum,value)>=0) {
        mpfr_sub(t[1],sum,t[0],RN); mpfr_add(t[1],t[1],value,RN);
    } else {
        mpfr_sub(t[1],value,t[0],RN); mpfr_add(t[1],t[1],sum,RN);
    }
    mpfr_add(correction,correction,t[1],RN); mpfr_set(sum,t[0],RN);
}
typedef struct {double magnitude;size_t index;} ordered_term;
static int compare_terms(const void *aa,const void *bb) {
    const ordered_term *a=aa,*b=bb;
    if (a->magnitude<b->magnitude)return -1;
    if (a->magnitude>b->magnitude)return 1;
    return (a->index>b->index)-(a->index<b->index);
}

/* mode 0: sequential, 1: magnitude-sorted, 2: pairwise,
   3: Neumaier, 4: Dot2, 5: magnitude-sorted Dot2. */
int rescued_readout(size_t n,size_t width,const double *phi,const double *weight,
                    double bias,unsigned p,unsigned mode,double *out) {
    size_t count=width+1;
    mpfr_t *a=vector(count,p),*b=vector(count,p),*prod=vector(count,p),*err=vector(count,p),*t=vector(8,p);
    ordered_term *order=malloc(count*sizeof(ordered_term));
    if (!a||!b||!prod||!err||!t||!order) {
        release(a,count);release(b,count);release(prod,count);release(err,count);release(t,8);free(order);return 1;
    }
    mpfr_t sum,corr,next,delta,split,tmp;
    mpfr_inits2(p,sum,corr,next,delta,split,tmp,(mpfr_ptr)0);
    mpfr_set_ui(split,1,RN);mpfr_mul_2ui(split,split,(p+1)/2,RN);mpfr_add_ui(split,split,1,RN);
    mpfr_set_ui(a[0],1,RN);mpfr_set_d(b[0],bias,RN);
    for(size_t j=0;j<width;++j)mpfr_set_d(b[j+1],weight[j],RN);
    for(size_t i=0;i<n;++i) {
        for(size_t j=0;j<width;++j)mpfr_set_d(a[j+1],phi[i*width+j],RN);
        for(size_t j=0;j<count;++j) {
            if(mode>=4)twoproduct(prod[j],err[j],a[j],b[j],split,t);
            else mpfr_mul(prod[j],a[j],b[j],RN);
            order[j].index=j;order[j].magnitude=fabs(mpfr_get_d(prod[j],RN));
        }
        if(mode==1||mode==5)qsort(order,count,sizeof(ordered_term),compare_terms);
        mpfr_set_zero(sum,0);mpfr_set_zero(corr,0);
        if(mode==2) {
            size_t active=count;
            while(active>1) {
                size_t k=0;
                for(size_t j=0;j+1<active;j+=2)mpfr_add(prod[k++],prod[j],prod[j+1],RN);
                if(active%2)mpfr_set(prod[k++],prod[active-1],RN);
                active=k;
            }
            mpfr_set(sum,prod[0],RN);
        } else for(size_t k=0;k<count;++k) {
            size_t j=order[k].index;
            if(mode>=4) {
                twosum(next,delta,sum,prod[j],t);
                mpfr_add(tmp,delta,err[j],RN);mpfr_add(corr,corr,tmp,RN);
                mpfr_set(sum,next,RN);
            } else if(mode==3)neumaier(sum,corr,prod[j],t);
            else mpfr_add(sum,sum,prod[j],RN);
        }
        if(mode>=3)mpfr_add(sum,sum,corr,RN);
        out[i]=mpfr_get_d(sum,RN);
    }
    release(a,count);release(b,count);release(prod,count);release(err,count);release(t,8);free(order);
    mpfr_clears(sum,corr,next,delta,split,tmp,(mpfr_ptr)0);
    return 0;
}

/* Construct from the center of each binomial row outwards. Mirror entries
   exactly, then merge stencils with compensated summation at p bits. */
int rescued_stencil(unsigned degree,const double *poly,const double *taylor,
                    double step,unsigned p,double *slopes,double *weights) {
    size_t width=2*degree+1;
    mpfr_t *sum=vector(width,p),*corr=vector(width,p),*t=vector(8,p);
    if(!sum||!corr||!t){release(sum,width);release(corr,width);release(t,8);return 1;}
    mpfr_t h,half,term,num,denom,ratio,index,other;
    mpfr_inits2(p,h,half,term,num,denom,ratio,index,other,(mpfr_ptr)0);
    mpfr_set_d(h,step,RN);mpfr_div_2ui(half,h,1,RN);
    int status=0;
    for(unsigned r=0;r<=degree;++r) {
        if(poly[r]==0.)continue;
        if(!isfinite(poly[r])||!isfinite(taylor[r])||taylor[r]==0.){status=2;break;}
        unsigned mid=r/2;
        mpfr_set_d(num,poly[r],RN);mpfr_set_d(denom,taylor[r],RN);mpfr_div(term,num,denom,RN);
        for(unsigned side=0;side<2;++side) {
            unsigned count=side? r-mid:mid;
            for(unsigned k=1;k<=count;++k) {
                mpfr_set_ui(index,k,RN);mpfr_mul(denom,h,index,RN);mpfr_div(term,term,denom,RN);
            }
        }
        if((r-mid)%2)mpfr_neg(term,term,RN);
        for(int j=(int)mid;j>=0;--j) {
            size_t left=degree+2*j-r,right=degree+r-2*j;
            neumaier(sum[left],corr[left],term,t);
            if(left!=right) {
                if(r%2)mpfr_neg(other,term,RN);else mpfr_set(other,term,RN);
                neumaier(sum[right],corr[right],other,t);
            }
            if(j>0) {
                mpfr_set_ui(num,j,RN);mpfr_set_ui(denom,r-j+1,RN);mpfr_div(ratio,num,denom,RN);
                mpfr_mul(term,term,ratio,RN);mpfr_neg(term,term,RN);
            }
        }
    }
    if(!status)for(size_t j=0;j<width;++j) {
        mpfr_set_si(index,(long)j-(long)degree,RN);mpfr_mul(num,half,index,RN);
        mpfr_add(term,sum[j],corr[j],RN);
        slopes[j]=mpfr_get_d(num,RN);weights[j]=mpfr_get_d(term,RN);
        if(!isfinite(weights[j])||mpfr_cmp_d(term,weights[j])||mpfr_cmp_d(num,slopes[j])){status=3;break;}
    }
    release(sum,width);release(corr,width);release(t,8);
    mpfr_clears(h,half,term,num,denom,ratio,index,other,(mpfr_ptr)0);
    return status;
}

/* Compensated coefficient sums and Taylor convolutions. Polynomial-basis
   recurrence stays p-bit, and each completed coefficient is one p-bit value. */
int rescued_polynomial_data(size_t n,unsigned degree,const double *nodes,const double *labels,
                           double bias,unsigned p,double *cheb_out,double *poly_out,double *taylor_out) {
    size_t size=degree+1;
    mpfr_t *coeff=vector(size,p),*cc=vector(size,p),*prev=vector(size,p),*cur=vector(size,p),
           *next=vector(size,p),*poly=vector(size,p),*pc=vector(size,p),*taylor=vector(size,p),*t=vector(8,p);
    if(!coeff||!cc||!prev||!cur||!next||!poly||!pc||!taylor||!t)return 1;
    mpfr_t x,y,t0,t1,t2,tmp,product,total,denom,two,one,bb,correction,error,delta,newtotal,split;
    mpfr_inits2(p,x,y,t0,t1,t2,tmp,product,total,denom,two,one,bb,correction,error,delta,newtotal,split,(mpfr_ptr)0);
    mpfr_set_ui(two,2,RN);mpfr_set_ui(one,1,RN);
    mpfr_set_ui(split,1,RN);mpfr_mul_2ui(split,split,(p+1)/2,RN);mpfr_add_ui(split,split,1,RN);
    for(size_t i=0;i<n;++i) {
        mpfr_set_d(x,nodes[i],RN);mpfr_set_d(y,labels[i],RN);mpfr_set(t0,one,RN);mpfr_set(t1,x,RN);
        neumaier(coeff[0],cc[0],y,t);
        if(degree){mpfr_mul(product,y,t1,RN);neumaier(coeff[1],cc[1],product,t);}
        for(unsigned k=2;k<=degree;++k) {
            mpfr_mul(tmp,two,x,RN);mpfr_mul(tmp,tmp,t1,RN);mpfr_sub(t2,tmp,t0,RN);
            mpfr_mul(product,y,t2,RN);neumaier(coeff[k],cc[k],product,t);
            mpfr_set(t0,t1,RN);mpfr_set(t1,t2,RN);
        }
    }
    mpfr_set_ui(denom,n,RN);
    for(unsigned k=0;k<=degree;++k) {
        mpfr_add(coeff[k],coeff[k],cc[k],RN);
        if(k)mpfr_mul(coeff[k],coeff[k],two,RN);
        mpfr_div(coeff[k],coeff[k],denom,RN);cheb_out[k]=mpfr_get_d(coeff[k],RN);
    }
    mpfr_set_ui(cur[0],1,RN);
    for(unsigned k=0;k<=degree;++k) {
        for(unsigned j=0;j<=degree;++j) {
            mpfr_mul(product,coeff[k],cur[j],RN);neumaier(poly[j],pc[j],product,t);
            mpfr_add(tmp,poly[j],pc[j],RN);poly_out[k*size+j]=mpfr_get_d(tmp,RN);
        }
        if(k==0) {
            for(unsigned j=0;j<=degree;++j)mpfr_set(prev[j],cur[j],RN);
            mpfr_set_zero(cur[0],0);if(degree)mpfr_set_ui(cur[1],1,RN);
        } else {
            for(unsigned j=0;j<=degree;++j) {
                if(j)mpfr_mul(tmp,two,cur[j-1],RN);else mpfr_set_zero(tmp,0);
                mpfr_sub(next[j],tmp,prev[j],RN);
            }
            for(unsigned j=0;j<=degree;++j){mpfr_set(prev[j],cur[j],RN);mpfr_set(cur[j],next[j],RN);}
        }
    }
    mpfr_set_d(bb,bias,RN);mpfr_tanh(taylor[0],bb,RN);
    for(unsigned r=0;r<degree;++r) {
        mpfr_set_zero(total,0);mpfr_set_zero(correction,0);
        for(unsigned j=0;j<=r;++j) {
            twoproduct(product,error,taylor[j],taylor[r-j],split,t);
            twosum(newtotal,delta,total,product,t);
            mpfr_add(tmp,error,delta,RN);mpfr_add(correction,correction,tmp,RN);mpfr_set(total,newtotal,RN);
        }
        mpfr_add(total,total,correction,RN);
        if(r==0)mpfr_sub(tmp,one,total,RN);else mpfr_neg(tmp,total,RN);
        mpfr_set_ui(denom,r+1,RN);mpfr_div(taylor[r+1],tmp,denom,RN);
    }
    for(unsigned r=0;r<=degree;++r)taylor_out[r]=mpfr_get_d(taylor[r],RN);
    release(coeff,size);release(cc,size);release(prev,size);release(cur,size);release(next,size);
    release(poly,size);release(pc,size);release(taylor,size);release(t,8);
    mpfr_clears(x,y,t0,t1,t2,tmp,product,total,denom,two,one,bb,correction,error,delta,newtotal,split,(mpfr_ptr)0);
    return 0;
}
