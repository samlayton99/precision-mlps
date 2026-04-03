# precision-mlps
This is a repo for a research project regarding machine epsilon precision mlps. We have already found an explicit construction that allows mlps to approximate functions to machine epsilon precision with computationally feasible designs, demonstrating theoretical capacity of MLPs. Now, the particular aim of this repo is to find an optimizer that can learn simple, single-variable functions through training, rather than through our construction.


# Hardware
uses cuda if available, then uses mps if available, and finally defaults to cpu. 

# Setup