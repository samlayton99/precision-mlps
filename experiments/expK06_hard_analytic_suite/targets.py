"""Frozen analytic targets, selected by function structure before baseline runs."""
import numpy as np
from scipy import special as sp

MAIN=['airy','kepler','elliptic_k','chirped_division']
EXTRA=['erf','chirp','localized','dawson','bessel0','bessel6','sine_integral','voigt','kummer','elliptic_e','fresnel','log_ratio']
NAMES=MAIN+EXTRA
LABELS={
'airy':r'$\mathrm{Ai}(-30x)$',
'kepler':r'$y-0.9\sin y=\pi x$',
'elliptic_k':r'$K\!\left(\frac{0.99}{1+9x^2}\right)$',
'chirped_division':r'$\frac{e^{-2x^2}\cos(30x+12x^2)}{1+100(x-0.2)^2}$',
'erf':r'$\mathrm{erf}(12(x-0.15))$',
'chirp':r'$\sin(8\pi(x+1)^2)$',
'localized':r'$e^{-40x^2}\sin(16\pi x)$',
'dawson':r'$D(20(x-0.1))$',
'bessel0':r'$J_0(60x)$',
'bessel6':r'$J_6(60x)$',
'sine_integral':r'$\mathrm{Si}(80x)$',
'voigt':r'$V(x-0.13;0.025,0.01)$',
'kummer':r'${}_1F_1(\frac12;1;-150x^2)$',
'elliptic_e':r'$E\!\left(\frac{0.999}{1+100(x+0.1)^2}\right)$',
'fresnel':r'$S(8x)$',
'log_ratio':r'$\frac{\log(1+400x^2)}{1+25(x-0.3)^2}$',
}
ENTIRE={'airy','erf','chirp','localized','dawson','bessel0','bessel6','sine_integral','voigt','kummer','fresnel'}

def kepler(x):
    m=np.pi*np.asarray(x,dtype=float)
    lo,hi=m-.9,m+.9
    y=m.copy()
    # Strictly monotone derivative >= .1; safeguarded Newton on the unique real branch.
    for _ in range(60):
        r=y-.9*np.sin(y)-m
        lo=np.where(r<0,y,lo);hi=np.where(r>0,y,hi)
        proposed=y-r/(1-.9*np.cos(y))
        y=np.where((proposed>=lo)&(proposed<=hi),proposed,(lo+hi)/2)
    return y

def target(name,x):
    x=np.asarray(x,dtype=float)
    if name=='airy':return sp.airy(-30*x)[0]
    if name=='kepler':return kepler(x)
    if name=='elliptic_k':return sp.ellipk(.99/(1+9*x*x))
    if name=='chirped_division':return np.exp(-2*x*x)*np.cos(30*x+12*x*x)/(1+100*(x-.2)**2)
    if name=='erf':return sp.erf(12*(x-.15))
    if name=='chirp':return np.sin(8*np.pi*(x+1)**2)
    if name=='localized':return np.exp(-40*x*x)*np.sin(16*np.pi*x)
    if name=='dawson':return sp.dawsn(20*(x-.1))
    if name=='bessel0':return sp.jv(0,60*x)
    if name=='bessel6':return sp.jv(6,60*x)
    if name=='sine_integral':return sp.sici(80*x)[0]
    if name=='voigt':return sp.voigt_profile(x-.13,.025,.01)
    if name=='kummer':return sp.hyp1f1(.5,1.,-150*x*x)
    if name=='elliptic_e':return sp.ellipe(.999/(1+100*(x+.1)**2))
    if name=='fresnel':return sp.fresnel(8*x)[0]
    if name=='log_ratio':return np.log1p(400*x*x)/(1+25*(x-.3)**2)
    raise KeyError(name)

def derivative(name,x):
    x=np.asarray(x,dtype=float)
    if name=='airy':return -30*sp.airy(-30*x)[1]
    if name=='kepler':return np.pi/(1-.9*np.cos(kepler(x)))
    if name in ['elliptic_k','elliptic_e']:
        z=x if name=='elliptic_k' else x+.1
        a,b=(.99,9) if name=='elliptic_k' else (.999,100)
        m=a/(1+b*z*z);dm=-2*a*b*z/(1+b*z*z)**2
        if name=='elliptic_k':return (sp.ellipe(m)/(1-m)-sp.ellipk(m))*dm/(2*m)
        return (sp.ellipe(m)-sp.ellipk(m))*dm/(2*m)
    if name=='chirped_division':
        a=np.exp(-2*x*x);p=30*x+12*x*x;d=1+100*(x-.2)**2
        return a*((-4*x*np.cos(p)-(30+24*x)*np.sin(p))/d-200*(x-.2)*np.cos(p)/d**2)
    if name=='erf':return 24/np.sqrt(np.pi)*np.exp(-(12*(x-.15))**2)
    if name=='chirp':return 16*np.pi*(x+1)*np.cos(8*np.pi*(x+1)**2)
    if name=='localized':return np.exp(-40*x*x)*(16*np.pi*np.cos(16*np.pi*x)-80*x*np.sin(16*np.pi*x))
    if name=='dawson':
        z=20*(x-.1);return 20*(1-2*z*sp.dawsn(z))
    if name in ['bessel0','bessel6']:return 60*sp.jvp(0 if name=='bessel0' else 6,60*x)
    if name=='sine_integral':return 80*np.sinc(80*x/np.pi)
    if name=='voigt':
        z=(x-.13+1j*.01)/(.025*np.sqrt(2))
        return (-2*z*sp.wofz(z)+2j/np.sqrt(np.pi)).real/(2*.025**2*np.sqrt(np.pi))
    if name=='kummer':return -150*x*sp.hyp1f1(1.5,2.,-150*x*x)
    if name=='fresnel':return 8*np.sin(np.pi*(8*x)**2/2)
    if name=='log_ratio':
        a=np.log1p(400*x*x);d=1+25*(x-.3)**2
        return 800*x/((1+400*x*x)*d)-50*(x-.3)*a/d**2
    raise KeyError(name)

def mp_target(name,x,mp):
    if name=='airy':return mp.airyai(-30*x)
    if name=='kepler':return mp.findroot(lambda y:y-mp.mpf('.9')*mp.sin(y)-mp.pi*x, float(kepler(float(x))))
    if name=='elliptic_k':return mp.ellipk(mp.mpf('.99')/(1+9*x*x))
    if name=='chirped_division':return mp.exp(-2*x*x)*mp.cos(30*x+12*x*x)/(1+100*(x-mp.mpf('.2'))**2)
    if name=='erf':return mp.erf(12*(x-mp.mpf('.15')))
    if name=='chirp':return mp.sin(8*mp.pi*(x+1)**2)
    if name=='localized':return mp.exp(-40*x*x)*mp.sin(16*mp.pi*x)
    if name=='dawson':
        z=20*(x-mp.mpf('.1'));return mp.sqrt(mp.pi)/2*mp.exp(-z*z)*mp.erfi(z)
    if name in ['bessel0','bessel6']:return mp.besselj(0 if name=='bessel0' else 6,60*x)
    if name=='sine_integral':return mp.si(80*x)
    if name=='voigt':
        z=(x-mp.mpf('.13')+1j*mp.mpf('.01'))/(mp.mpf('.025')*mp.sqrt(2))
        return mp.re(mp.exp(-z*z)*mp.erfc(-1j*z))/(mp.mpf('.025')*mp.sqrt(2*mp.pi))
    if name=='kummer':return mp.hyp1f1(mp.mpf('.5'),1,-150*x*x)
    if name=='elliptic_e':return mp.ellipe(mp.mpf('.999')/(1+100*(x+mp.mpf('.1'))**2))
    if name=='fresnel':return mp.fresnels(8*x)
    if name=='log_ratio':return mp.log(1+400*x*x)/(1+25*(x-mp.mpf('.3'))**2)
    raise KeyError(name)
