"""Exact GELU, native GD, and bias-corrected Adam, all in float64."""
from __future__ import annotations
import torch


@torch.jit.script
def gelu_gradient(theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    width=(theta.numel()-1)//3
    a,b,c=theta[:width],theta[width:2*width],theta[2*width:3*width]
    z=a[:,None]*x[None,:]+b[:,None]
    # Reuse the normal CDF in the activation and its analytic derivative.
    cdf=.5*(1.+torch.erf(z*0.7071067811865475244))
    features=z*cdf
    residual=torch.matmul(c,features)+theta[-1]-y
    weighted=residual/x.numel()
    derivative=cdf+z*torch.exp(-.5*z*z)*0.39894228040143267794
    dh=derivative*c[:,None]*weighted[None,:]
    gradient=torch.cat((torch.sum(dh*x[None,:],dim=1),torch.sum(dh,dim=1),
                        torch.matmul(features,weighted),torch.sum(weighted).reshape(1)))
    return gradient,.5*torch.mean(residual*residual)


@torch.jit.script
def train_chunk(theta: torch.Tensor, first: torch.Tensor, second: torch.Tensor,
                x: torch.Tensor, y: torch.Tensor, lr: float, count: int, start_step: int,
                adam: bool, beta1: float=.9, beta2: float=.999, epsilon: float=1.e-8
                ) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,int,bool]:
    for k in range(count):
        gradient,loss=gelu_gradient(theta,x,y)
        if not bool(torch.isfinite(loss)) or not bool(torch.all(torch.isfinite(gradient))):
            return theta,first,second,k,False
        if adam:
            m=beta1*first+(1.-beta1)*gradient
            v=beta2*second+(1.-beta2)*gradient*gradient
            step=start_step+k+1
            denom=torch.sqrt(v/(1.-beta2**step))+epsilon
            candidate=theta-(lr/(1.-beta1**step))*m/denom
        else:
            m,v=first,second
            candidate=theta-lr*gradient
        if not bool(torch.all(torch.isfinite(candidate))):
            return theta,first,second,k,False
        theta,first,second=candidate,m,v
    return theta,first,second,count,True


@torch.jit.script
def predict(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    w=(theta.numel()-1)//3
    z=theta[:w,None]*x[None,:]+theta[w:2*w,None]
    return torch.matmul(theta[2*w:3*w],torch.nn.functional.gelu(z,approximate="none"))+theta[-1]
