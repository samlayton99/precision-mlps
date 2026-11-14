"""Float64 tanh Adam with simultaneous gradients and persistent bias-corrected moments."""
import torch

@torch.jit.script
def train_chunk(theta: torch.Tensor, first: torch.Tensor, second: torch.Tensor,
                x: torch.Tensor, y: torch.Tensor, lr: float, count: int, start_step: int
                ) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,int,bool]:
    w=(theta.numel()-1)//3
    for k in range(count):
        a,b,c=theta[:w],theta[w:2*w],theta[2*w:3*w]
        h=torch.tanh(a[:,None]*x[None,:]+b[:,None])
        e=torch.matmul(c,h)+theta[-1]-y
        weighted=e/x.numel()
        dh=(1-h*h)*c[:,None]*weighted[None,:]
        g=torch.cat((torch.sum(dh*x[None,:],dim=1),torch.sum(dh,dim=1),
                     torch.matmul(h,weighted),torch.sum(weighted).reshape(1)))
        m=.9*first+.1*g
        v=.999*second+.001*g*g
        step=start_step+k+1
        candidate=theta-(lr/(1.-.9**step))*m/(torch.sqrt(v/(1.-.999**step))+1.e-8)
        if not bool(torch.all(torch.isfinite(candidate))):
            return theta,first,second,k,False
        theta,first,second=candidate,m,v
    return theta,first,second,count,True
