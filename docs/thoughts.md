## Use lstsq over QI
- So basically, I verified the results numerically, the machinery works.
- I found that lstsq gives much better numerical results on all fronts, 
so I tested to see if they were the same solution... they aren't, but if you project the QI solution into the null space of phi, then they are identical. so the phi matrix has super big null space and weirdly conditioned, but that conditioning is not correlated with the ability to get good precision. but it basically chooses the same solution in the null space 
- norm of weight also drops down, the norm solution is pretty solid and not a big deal.

## it is robust to noise and it scales well
- noise in x is prety harmless
- noise in y is much harder, follows statistics 1/sqrt(n)
- noise in centers is painful
- as we scale the size and data, there are precision scaling laws for both.

## we have 3 conditions that make the geometry work well
### Gamma
- gamma in the right regime is really important
- ideal lambda stays pretty constant as you increase frequency and across function types. matters about 8-10 orders of precision

### bias uniformity
- correct center placement gives you about 2 orders of magnitude of precision. pretty useful to get this right
- more uniformity is giving wider lambda error valleys, (and often deeper) across all functions
- uniformity improves precision for almost every lambda and function choice, gradual improvements the entire time, but most dramatic improvements on the last nudge to uniformity (aliasing?)
- 

### weight identicalness
weights, place it on the unit l1 ball, interpolate it across the surface to 



### Interpolating geometries
Interpolating between, we see some interesting things

note that if we interpolate weights through the center, then we get a lot of dead neurons, so weights can't be too close to zero



odd point: runge had a defect where for small N, uniformity hurt. this was the only case. in this same case, we were able to get machine epsilon precision if the biases were more compact and less spread out. this is something further to test. 





So one thing we can do is fine tune the lambda, it matters, but maybe reparameterizing and letting adam go at it can put it in the optimal regime


compare 2nd order, 1st order, lstsq, toeplitz




