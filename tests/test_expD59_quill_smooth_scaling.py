import numpy as np
from docs.qi_practical_guide.reference import fit_qi,default_halo,MLP
from experiments.expD59_quill_smooth_scaling.run import load_config,values,target,audit_points
from experiments.expD59_quill_smooth_scaling.audit_scalar import evaluate_scalar_columns


def test_smooth_subset_and_shared_fit(tmp_path):
    config=load_config()
    assert len(config["targets"])==20 and "smooth_bump" in config["targets"]
    assert not set(config["excluded"]) & set(config["targets"])
    names=["square","sine_4pi","smooth_bump"]
    x=np.linspace(-1,1,257);xe=np.linspace(-.999,.999,333)
    y=values(names,x,np.ones(3))
    model,info=fit_qi(x,y,directions=np.ones((1,1)),intervals=32,lam=.25,rcond=1e-14)
    assert info["width"]==32+1+2*default_halo(32)
    for i,name in enumerate(names):
        scalar=MLP(model.weight,model.hidden_bias,model.readout[:,i:i+1],model.output_bias[i:i+1],True,"tanh")
        np.testing.assert_array_equal(evaluate_scalar_columns(model,xe)[:,i],scalar(xe))
        # The coarse smooth-bump fit has ~2.5e10 readout mass; tiny absolute
        # agreement between separate ill-conditioned solves is not a valid test.
        if name!="smooth_bump":
            independent,_=fit_qi(x,y[:,i],directions=np.ones((1,1)),intervals=32,lam=.25,rcond=1e-14)
            np.testing.assert_allclose(scalar(xe),independent(xe),atol=2e-11,rtol=2e-11)
    path=tmp_path/"model.npz";model.save(path)
    np.testing.assert_array_equal(MLP.load(path)(xe),model(xe))
    assert np.all(audit_points()>=-1) and np.all(audit_points()<=1)
