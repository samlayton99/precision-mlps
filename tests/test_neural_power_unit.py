import math
import numpy as np
import torch
from torch.func import functional_call

from src.models.neural_power_unit import NeuralPowerUnit, NPURegressor
from experiments.expK02_inverse_sqrt_extension.construction import Extension
from experiments.expK03_npu_inverse_sqrt.run import halo_nodes, evaluation_grid, relative_l2


def test_published_complex_formula_and_gradients():
    model = NeuralPowerUnit(2, 2)
    with torch.no_grad():
        model.real.copy_(torch.tensor([[.2, -.7], [.6, -.3]]))
        model.imag.copy_(torch.tensor([[.1, .2], [-.2, .3]]))
        model.gate.copy_(torch.tensor([.3, .8]))
    x = torch.tensor([[-.3, .8], [.2, -.9], [.7, .4]], dtype=torch.float64)
    re, im, g = [p.detach().numpy() for p in [model.real, model.imag, model.gate]]
    a = x.numpy()
    radius = g * (np.abs(a) + np.finfo(float).eps) + 1 - g
    angle = np.maximum(-np.sign(a), 0) * np.pi * g
    expected = np.exp(np.log(radius)@re.T-angle@im.T)*np.cos(np.log(radius)@im.T+angle@re.T)
    np.testing.assert_allclose(model(x).detach(), expected, rtol=5e-15, atol=1e-15)
    names, params = zip(*model.named_parameters())
    assert torch.autograd.gradcheck(lambda *p: functional_call(model, dict(zip(names,p)), (x,)), params)


def test_inverse_sqrt_is_representable_by_one_real_power():
    model = NeuralPowerUnit(1, 1, complex_weights=False)
    with torch.no_grad():
        model.real.fill_(-.5)
        model.gate.fill_(1.)
    e = Extension()
    assert relative_l2(model, evaluation_grid(e, 256), e.eps) < 1e-12
    x = torch.logspace(-4, 0, 100, dtype=torch.float64)[:,None]
    torch.testing.assert_close(model(x), (x+torch.finfo(x.dtype).eps).pow(-.5), rtol=1e-15, atol=1e-14)


def test_positive_inputs_zero_imaginary_branch_matches_realnpu():
    npu, real = NPURegressor(seed=0), NPURegressor(complex_weights=False,seed=0)
    x = torch.linspace(.0001,1,100,dtype=torch.float64)[:,None]
    torch.testing.assert_close(npu(x),real(x),rtol=0,atol=0)
    npu(x).square().sum().backward()
    assert torch.count_nonzero(npu.power.imag.grad) == 0
    assert sum(p.numel() for p in npu.parameters())==19
    assert sum(p.numel() for p in real.parameters())==13


def test_forward_gate_clipping_has_reference_zero_gradient_outside_bounds():
    model = NeuralPowerUnit(2,1)
    with torch.no_grad():
        model.gate.copy_(torch.tensor([-.2,1.2]))
    model(torch.tensor([[.2,-.7]],dtype=torch.float64)).sum().backward()
    assert torch.count_nonzero(model.gate.grad)==0


def test_halo_matches_saved_qi_geometry_and_extension():
    e=Extension()
    x=halo_nodes(e,256,32)
    assert len(x)==4815
    np.testing.assert_allclose([x.min(),x.max()],[-.4096,3.6864],atol=1e-15)
    z=-1+2*np.arange(-32,289)/256
    expected=np.concatenate([s*(2+2*z) for s in e.scales])
    np.testing.assert_array_equal(x,expected)
    y=e.value(x)
    assert np.isfinite(y).all()
    inner=(x>=e.eps)&(x<=1)
    np.testing.assert_allclose(y[inner],x[inner]**-.5,rtol=3e-16)


def test_evaluation_metric_uses_same_physical_l2():
    class Perturbed(torch.nn.Module):
        def forward(self,x): return 1.01*x.pow(-.5)
    e=Extension()
    np.testing.assert_allclose(relative_l2(Perturbed(),evaluation_grid(e,64),e.eps),.01,atol=1e-16)


def test_checkpoint_adam_continuation_is_identical():
    import copy
    model=NPURegressor(seed=3)
    opt=torch.optim.Adam(model.parameters(),lr=.001,foreach=False)
    x=torch.linspace(.0001,1,100,dtype=torch.float64)[:,None]
    def step(m,o):
        o.zero_grad();(m(x)-x.pow(-.5)).square().mean().backward();o.step()
    for _ in range(3): step(model,opt)
    clone=NPURegressor(seed=4);clone.load_state_dict(copy.deepcopy(model.state_dict()))
    other=torch.optim.Adam(clone.parameters(),lr=.001,foreach=False)
    other.load_state_dict(copy.deepcopy(opt.state_dict()))
    for _ in range(2):step(model,opt);step(clone,other)
    for a,b in zip(model.parameters(),clone.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_adam_mlp_matches_quill_architecture_and_centering():
    from src.models.centered_tanh import CenteredTanhMLP
    from experiments.expK02_inverse_sqrt_extension.construction import Network
    model=CenteredTanhMLP(7,seed=0)
    assert sum(p.numel() for p in model.parameters())==3*7+1
    state=model.state_dict()
    reference=Network(state['net.0.weight'].numpy().ravel(),state['net.0.bias'].numpy(),
                      state['net.2.weight'].numpy().ravel(),float(state['net.2.bias'][0]),1e-4,'test')
    x=np.r_[[-.4096,0.],np.geomspace(1e-4,1,100),[3.6864]]
    np.testing.assert_allclose(model(torch.from_numpy(x[:,None])).detach().numpy().ravel(),
                               reference.predict(x),rtol=1e-14,atol=1e-15)
    assert sum(p.numel() for p in CenteredTanhMLP(4815).parameters())==14446


def test_chunked_metric_matches_direct_evaluation():
    from experiments.expK03_npu_inverse_sqrt.run import predict
    from src.models.centered_tanh import CenteredTanhMLP
    model=CenteredTanhMLP(31)
    x=torch.linspace(-.4,3.7,2100,dtype=torch.float64)[:,None]
    torch.testing.assert_close(predict(model,x,batch=128),model(x),rtol=1e-14,atol=1e-15)


def test_train_saves_validation_optimum_after_training_starts(tmp_path):
    import json
    from pathlib import Path
    from experiments.expK03_npu_inverse_sqrt.run import train
    config=json.loads((Path(__file__).resolve().parents[1]/'experiments/expK03_npu_inverse_sqrt/config.json').read_text())
    config.update(width=2,steps=3,log_every=1,checkpoint_every=3,minimum_selection_step=2)
    row=train(config,'RealNPU','matched_halo',tmp_path)
    assert row['best_trained_validation_step'] in [2,3]
    checkpoint=torch.load(tmp_path/'RealNPU_matched_halo_best_trained_validation.pt',weights_only=False)
    model=NPURegressor(2,complex_weights=False);model.load_state_dict(checkpoint['model'])
    e=Extension()
    assert relative_l2(model,evaluation_grid(e,256),e.eps)==row['best_trained_validation_test_rel_l2']


def test_adam_width_selection_uses_validation_not_test_scores():
    from experiments.expK03_npu_inverse_sqrt.adam_tuning import select_widths
    rows=[{'hidden_units':h,'learning_rate':lr,'best_trained_validation_rel_l2':score,
           'best_trained_validation_test_rel_l2':1/score,'best_trained_validation_step':7500}
          for h,lr,score in [(8,.01,.4),(8,.001,.2),(16,.01,.3),(32,.01,.1),(64,.01,.5)]]
    widths,ranking=select_widths(rows)
    assert widths==[8,16,32]
    assert [r['width'] for r in ranking]==[32,8,16,64]
