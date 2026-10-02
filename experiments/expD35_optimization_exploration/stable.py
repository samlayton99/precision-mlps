"""Cancellation-aware neighboring features, with their exact analytic derivative."""
import jax
import jax.numpy as jnp
from . import core


def difference(a,b,xp=jnp):
    d=a-b;absolute=xp.abs(d)
    return (2*xp.sign(d)*xp.exp(absolute-xp.abs(a)-xp.abs(b))*(-xp.expm1(-2*absolute))
            /((1+xp.exp(-2*xp.abs(a)))*(1+xp.exp(-2*xp.abs(b)))))


@jax.custom_jvp
def tanh_difference(a,b):
    return difference(a,b)


@tanh_difference.defjvp
def derivative(primals,tangents):
    a,b=primals;da,db=tangents
    return tanh_difference(a,b),core.old.sech_squared(a)*da-core.old.sech_squared(b)*db


def predict(z,g,x):
    q=z[1:g.width+1]*jnp.asarray(g.alpha[1:].cumsum())
    gamma=z[g.width+1:2*g.width+1]/g.h
    pre=(x[:,None]-jnp.asarray(g.centers))*gamma
    features=tanh_difference(pre[:,:-1],pre[:,1:])
    return z[0]*g.alpha[0]+features@q[:-1]+core.old.tanh(pre[:,-1])*q[-1]
