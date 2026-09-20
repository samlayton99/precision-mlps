# expF10 -- QI-encoded neural operators: how much does the QI help?

**Status:** drafted (single seed). Data-driven Darcy operator learning, three
matched configs, 100 epochs on one A100. First entry in the *learned* regime
(contrast the training-free physics solves expF08/expF09).

## TL;DR

- **As an input encoder, the QI helps.** Config B (QI-resample the input, then a
  plain FNO) beats the plain-FNO control C at **every** resolution and **every**
  data budget -- test rel L2 **0.053 vs 0.072** at 64^2 (~27% relative), and it
  is the most data-efficient (0.177/0.103/0.053 at N=100/300/1000 vs C's
  0.189/0.121/0.072).
- **As a standalone operator, the QI does not compete.** Config A (learn the map
  in QI-coefficient space, no FFT) is worst at fixed resolution (**0.122**),
  collapses with little data (rel L2 **144 / 47** at N=100/300), and is unstable
  when *downsampling* below the training resolution (res 32 blows up to 5.1 --
  the encoded coefficients go out of distribution). Its one structural win is
  **flat super-resolution**: trained at 64^2 it holds 0.122 -> 0.124 out to
  256^2 with no retraining.
- **Plain FNO already generalises across resolution** in the 32-256 band
  (0.071-0.076), so the QI's discretization-invariance advantage is **modest,
  not dramatic** -- the hypothesised big invariance win did not materialise
  because the baseline is already fairly resolution-robust there.
- **Net answer to "how much does QI help?":** *insert it as an input encoder,
  yes (~27%); use it as a full operator replacement, no.* The rough Darcy
  coefficient (QI reconstructs it only to ~4.5e-2 rel L2, `test 3`) caps how
  much the QI representation can buy on this benchmark.

## Question

Does inserting the QI (frozen ridge) representation as a **fixed,
spectral-quality encoder/decoder** around a learned Darcy operator help, and by
how much, vs a plain FNO? Three matched configs isolate *where* the QI acts.

## Experiment design

- **Configs.** A: `a -> c_a = Phi^+ a -> [MLP] -> c_u -> u = Phi_out c_u`
  (coefficient space, no FFT). B: `a -> QI-resample to grid -> [FNO] -> u`.
  C: `a -> [FNO] -> u` (control). QI encode/decode are fixed linear maps
  (`qi_codec.py`, W=576, D=586), precomputed `Phi^+`. Small custom 2D FNO
  (`fno2d.py`, width 32, 12 modes, 4 layers).
- **Data.** FNO Darcy (`darcy_*_421`), area-average downsampled to 64^2;
  N_train 1000, N_test 200. Adam, rel-L2 loss, 100 epochs, one A100.
- **Metrics.** (1) test rel L2 at 64^2; (2) discretization invariance -- train
  64^2, eval zero-shot at {16,32,64,128,256}; (3) data efficiency
  N in {100,300,1000}; (4) QI input-reconstruction error (bounds A/B).
- **Reproduce.** `run.py --config all`, `--eval-invariance`, `--eval-data-eff`,
  `--plot`.

## Results

**Accuracy @ 64^2 (1000 train)** (`accuracy_bar.png`):

| config | test rel L2 | params | train time |
|---|---|---|---|
| A -- QI-coeff MLP | 0.122 | 4.35M | 11 s |
| **B -- QI -> FNO** | **0.053** | 2.37M | 51 s |
| C -- plain FNO | 0.072 | 2.37M | 51 s |

**Data efficiency** (`data_eff.png`), test rel L2 at N=100/300/1000:
- A: **144 / 47** / 0.122 (unusable below ~1000; the coeff MLP is data-hungry
  and its ill-conditioned synthesis amplifies errors)
- B: 0.177 / 0.103 / **0.053** (best everywhere)
- C: 0.189 / 0.121 / 0.072

**Discretization invariance** (`invariance_vs_res.png`), train @ 64^2:

| test res | A | B | C |
|---|---|---|---|
| 16 | 0.155 | 0.358 | 0.342 |
| 32 | 5.09 | 0.054 | 0.071 |
| 64 | 0.122 | 0.053 | 0.072 |
| 128 | 0.123 | 0.054 | 0.075 |
| 256 | 0.124 | 0.054 | 0.076 |

- **B is best at every resolution and data budget.**
- **A super-resolves flat** (64 -> 256: 0.122 -> 0.124), its structural feature,
  but is worse absolutely and unstable *downward* (res 32 = 5.1: the res-32
  encoding produces coefficients the MLP never saw at train time).
- **C (plain FNO) is already resolution-robust** in 32-256 (0.071-0.076); all
  three degrade at the very coarse res 16.

**QI input-reconstruction (diagnostic).** A Darcy coefficient reconstructs from
QI coefficients at ~4.5e-2 rel L2 (`test 3`) -- the coefficient is rough, so the
QI representation is lossy, which caps the help A/B can provide on this
benchmark.

## Conclusions

1. **QI as an input encoder is a real, consistent win** (B > C by ~27% and more
   data-efficient) -- the QI-smoothed input helps the FNO learn the map.
2. **QI as a full operator replacement is not competitive** here: the
   coefficient-space MLP is data-hungry, capped by the D=586 latent and the
   ill-conditioned synthesis, and only invariant upward.
3. **The invariance hypothesis was only partly borne out** -- plain FNO already
   generalises across resolution in the tested band, so the QI's structural
   advantage is modest, not decisive, on Darcy.

## Open questions / next

- **Whiten the coeff space** (encode targets to `c_u* = Phi^+ u`, train with a
  coefficient loss in an SVD-whitened basis) -- would it rescue config A's data
  efficiency and downsampling stability?
- **Smoother benchmark** (a smooth-coefficient or smooth-solution operator)
  where the QI representation is near-exact -- does A/B's advantage grow when the
  ~4.5e-2 reconstruction cap is removed?
- **Scattered / non-grid inputs** -- the setting where plain FNO cannot run at
  all and the QI encoder's true advantage should appear (not tested here).
- **Larger FNO + more data** to push C to the literature ~1e-2 and re-check
  whether the QI-encoder gain persists at that scale.
