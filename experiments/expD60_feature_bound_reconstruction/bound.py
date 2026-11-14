"""Direct B-coordinate transcription of QUILL Appendix F.4--F.7."""
import math
import numpy as np


def coefficients(x):
    u2=x*x-np.mean(x*x);u3=x**3-np.mean(x**4)/np.mean(x*x)*x
    s2=float(np.mean(u2*u2));s3=float(np.mean(u3*u3))
    alpha={3:1/3,5:2/15,7:17/315}
    A={p:alpha[p]*2**(p/2)*p**(p/2)/(p+1)**((p+1)/2) for p in alpha}
    return A,{p:(p+1)*A[p] for p in A},math.sqrt(s2/64+3*s3/256),math.sqrt(8*s2/27+3*s3/16)


def evaluate(history,x,width,lr,reference,start=25000,family="refined"):
    """Empirical reconstruction using interpolated saved structural allowances."""
    steps=history['step'].astype(int);first=int(np.flatnonzero(steps==start)[0])
    A,C,B3,D3=coefficients(x);W=float(width)
    E0=float(np.max(history['nonaffine_residual_rms'][first:]))
    t3=float(history['tau_3'][first]);t5=float(history['tau_5'][first])
    target=float(history['target_nonaffine_rms'][first])
    B=math.sqrt(history['parameter_energy'][first])
    moments=np.column_stack([history[f'chi_{p}'] for p in [4,6,10,14]])
    choices={'generic':(0,),'projected':(0,1),'refined':(0,1,2,3)}[family]
    counts=[0]*4;crossing=None;violations=0;trace=[]
    def allowances(norm,chi):
        S={p:chi[i]*norm**p/W**(p/2-1) for i,p in enumerate([4,6,10,14])}
        Q=min(A[3]*S[4],B3*S[4]+A[5]*S[6])
        r6,r10,r14=(math.sqrt(S[p]) for p in [6,10,14])
        V=(C[3]*E0*r6,
           E0*(D3*r6+C[5]*r10),
           D3*r6*(t3+Q)+C[5]*E0*r10,
           D3*r6*t3+C[5]*r10*t5+(D3*r6+C[5]*r10)*Q+C[7]*E0*r14)
        return Q,V
    Q,_=allowances(B,moments[first]);trace.append([start,B/math.sqrt(W),max(0,target-Q)])
    for segment in range(first,len(steps)-1):
        n0,n1=int(steps[segment]),int(steps[segment+1]);dt=n1-n0
        chi0=moments[segment];delta=(moments[segment+1]-chi0)/dt
        r0=float(history['tracking_norm'][segment]);rd=float(history['tracking_norm'][segment+1]-r0)/dt
        for j in range(dt):
            chi=chi0+j*delta;Q,V=allowances(B,chi)
            selected=min(choices,key=V.__getitem__);counts[selected]+=1
            B+=lr*(V[selected]+r0+j*rd)
            if B/math.sqrt(W)>=reference:
                crossing=n0+j+1
                trace.append([crossing,B/math.sqrt(W),0.]);break
        if crossing is not None:break
        Q,_=allowances(B,moments[segment+1])
        trace.append([n1,B/math.sqrt(W),max(0,target-Q)])
        violations+=int(B+1e-9<math.sqrt(history['parameter_energy'][segment+1]))
    return {'family':family,'start_step':start,'end_step':int(steps[-1]),'E0':E0,
        'initial_energy':float(history['parameter_energy'][first]),'initial_rms_slope':float(history['gamma_rms'][first]),
        'end_rms_slope':float(history['gamma_rms'][-1]),'reference':reference,'reference_crossing':crossing,
        'below_reference':crossing is None,'end_ceiling':float(B/math.sqrt(W)) if crossing is None else None,
        'end_error_floor':float(max(0,target-Q)) if crossing is None else 0.,
        'force_choices':counts,'parameter_envelope_violations':violations,'trace':trace}
