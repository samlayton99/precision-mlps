import numpy as np
import torch
from experiments.expD56_scale_acquisition.model import initial_parameters, midpoint_grid, target
from experiments.expD57_gelu_scale_acquisition.model import gelu_gradient, train_chunk, predict
from experiments.expD57_gelu_scale_acquisition.run import load_config, cases


def test_exact_gelu_gradient_matches_autograd():
    torch.set_num_threads(1)
    theta=initial_parameters(17,30,4.)
    x=torch.from_numpy(midpoint_grid(65))
    y=torch.from_numpy(target("runge_100",x.numpy()))
    p=theta.clone().requires_grad_(True)
    z=x[:,None]*p[:17]+p[17:34]
    output=torch.nn.functional.gelu(z,approximate="none")@p[34:51]+p[-1]
    loss=.5*(output-y).square().mean()
    loss.backward()
    actual,actual_loss=gelu_gradient(theta,x,y)
    torch.testing.assert_close(actual,p.grad,rtol=1e-11,atol=1e-13)
    torch.testing.assert_close(actual_loss,loss)
    torch.testing.assert_close(predict(theta,x),output)


def test_gd_and_adam_match_pytorch_and_exact_resume():
    torch.set_num_threads(1)
    x=torch.from_numpy(midpoint_grid(64)); y=torch.from_numpy(target("sine_4pi",x.numpy()))
    for use_adam in (False,True):
        start=initial_parameters(17,30,2.)
        zeros=torch.zeros_like(start)
        actual,m,v,n,ok=train_chunk(start,zeros,zeros,x,y,.001,51,0,use_adam)
        assert ok and n==51
        parameter=start.clone().requires_grad_(True)
        optimizer=(torch.optim.Adam([parameter],lr=.001,betas=(.9,.999),eps=1e-8,foreach=False)
                   if use_adam else torch.optim.SGD([parameter],lr=.001))
        for _ in range(51):
            optimizer.zero_grad()
            z=x[:,None]*parameter[:17]+parameter[17:34]
            output=torch.nn.functional.gelu(z,approximate="none")@parameter[34:51]+parameter[-1]
            (.5*(output-y).square().mean()).backward()
            optimizer.step()
        torch.testing.assert_close(actual,parameter,rtol=1e-10,atol=1e-12)
        p1,m1,v1,_,_=train_chunk(start,zeros,zeros,x,y,.001,19,0,use_adam)
        # Exercise the same tensor -> array -> tensor conversion as checkpoint resume.
        state=[torch.from_numpy(t.numpy().copy()) for t in (p1,m1,v1)]
        p2,m2,v2,_,_=train_chunk(*state,x,y,.001,32,19,use_adam)
        for a,b in ((actual,p2),(m,m2),(v,v2)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_matched_matrix():
    config=load_config()
    assert len(cases(config))==36
    assert config["width"]==196 and config["steps"]==200000
    assert set(c[1] for c in cases(config))=={"sine_4pi","runge_100"}
    for multiplier in config["initialization_multipliers"]:
        p=initial_parameters(196,config["seed"],multiplier).numpy()
        old=f"results/checkpoint_D_optimizers/expD56_scale_acquisition/sweep/runs/sine_4pi__init{multiplier:g}__lr1e-02/initial.npz"
        with np.load(old) as saved: np.testing.assert_array_equal(p,saved["theta"])
