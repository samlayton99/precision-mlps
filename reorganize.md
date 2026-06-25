(first job)

okay first I want you to go in and reorganize all of my results. look at each of the small results.md (and the global one), and try and simplify. the problem is consistent for most of them: I am skimming over most of it, and not taking it in, so it needs to cut out the things that are less relevant. Just take a fresh look at each of the results page, look at the individual figures, what is currently written, and then data, and see if there are things that are less relevant that should be cut, or things that could be compressed. don't loose valuable information, just see if you can trim the fat. I wrote comments at the bottom of most of them for direction.
also please include a very clear TLDR and what questions/hypothesis these experiments are answering section. they should all have this of some form. Don't forget to validate what is written with the actual results in the folder.

One thing that is persistent, there are just too many numbers in the writeup (paragraphs are full of N=128, has error 1.23x10^-12, N=256, ...). It makes a lot of the prose nearly impossible to parse and read. There needs to be a better balance. Think of using raw numbers like currency, spend them only where it matters most (that doesn't mean be stingy with them, but overusing them makes them not have value at all.)

the second thing, each writeup should have the same structure. think about that structure, how to improve it, and then make it follow that same format. there should be an additional details section that allows the writeup results.md to be customizable to whatever that experiment needs. update the claude.md for future results.md's that are written to share these conventions and format that I have written. 

After going through and updating the individual experiments (2nd job)

We also should rename all of the experiments and regroup them so that it makes more sense. more descriptive name, better groupings. 

Here are some of my thoughts on the structure of the reorganization

We should also rename it by grouping, so grouping expA01 would be first experiment of checkpoint A, expA02 is likewise, and expC02 would be 2 of checkpoint C. 

The folder structure should also be 
results/
    checkpoint_A/
        expA01_someexp...
        expA02
        ...
    checkpoint_B/


What experiments belong in each (currently using the old naming convention, so don't get confused, you will update it to the new one):

Checkpoint A — We choose the right numerical validation, and justify our methods 
- exp01_numerics_sanity (basically verifies that our toolset works)
- exp03_qi_vs_lstsq (verifies that lstsq is numerically superior)
- exp05_activation_coeff (more details about the matrices, a little bit of a tangent)
- exp04_coeff_nullspace (gives some theoretical support to lstsq, as they are the same solution minus a nullspace)
to include: a study of the norm/magnitude of the solved readout coefficients

Checkpoint B — We show it works practically with scaling laws and robust to noise. 
- exp08 sampling_and_noise (we jitter the x sample points, the centers and the y points. very clear that jittering centers is sensitive, x is not, y is sensitive but can get rid of according to statistics)
- exp09 scaling laws (basically as we scale the number of datapoints and the width, with varying noise, precision improvements follow a scaling law)


Checkpoint C — how important is choosing the right geometry
- exp02 lambda_tradeoff (validates the U exists predicted in the theory, isolates regime, and shows it holds both QI and lstsq)
- exp06 lambda_vs_frequency(this shows it is constant for higher and higher frequency)
- exp18_lambda_basin (this is an even deeper dive into finding the right lambda regimes, finds its all constants)
- exp07_center_geometry (finds that geometry matters to be uniform)
- exp16 geometry interpolations (shows the interaction between gamma, weight identicalness, and bias uniformity)

Checkpoint D — how well do optimizers find the geometry. 
- exp12_geometry_ladder
- exp17_adam_geomtry 


Checkpoint E — Extending this to 2d
- exp11_geometry_zoo_2d (we can delete exp10, it is a subset of exp11)


Make sure all open questions are placed at the bottom of each checkpoint. each experiment should have an open questions section, but this should be conservative (not too many) and the checkpoint should aggregate the open questions at the bottom of its section.

So the stages of implementation look like this:
1) go in and fix the individual results.md with the changes I have approved. 
2) organize the file structure as I have given it
3) rename the different experiments to the right convention and better descriptive
4) rewrite the overall results.md file that summarizes and compiles everything. also parse out future experiments to where they need to go in which checkpoints. 
5) go through and match it with the experiments folder, making the code, filepath, and everything so that new experiments or rerun experiments won't break. filepath references need to work. 
6) do a final check/sweep with new eyes to make sure that there are no incorrect, old regime, hanging pointers. make sure dependencies, file paths, and everything is organized correctly. 


Questions for you before we begin:
1) should we fold in exp10 with exp11, i.e. delete 10 entirely? 
2) Is exp12 still relevant? i feel like we basically did that with the exp17. 
3) how many of the proposed experiments are still relevant now? (I think less)


(here are some independent thoughts, they probably don't go here, but I will put them here)
- I see two remaining things: both depth, and functions going from R^n to R^m. We have already solved R to R^m. so this is the next work.
- it seems that this initialization works great when it is over a relevant domain, so domain is relevant here
- what if in a transformer, we initialize the first hidden layers like this? we would have to figure out depth, domain, and higher dim here for that to happen first.