# pystream

An MOA-based implementation for data stream classification in Python/Cython

## Includes:
  
  __Base learners__:
  - Very Fast Decision Tree (https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf)
  - Strict Very Fast Decision Tree (https://arxiv.org/abs/1805.06368)
  
  __Ensembles__:
  - OzaBag/OzaBoost (https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf)
  - OAUE (https://www.sciencedirect.com/science/article/pii/S0020025513008554)
  - LevBag (https://core.ac.uk/download/pdf/129931682.pdf)
  - ARF (https://link.springer.com/article/10.1007/s10994-017-5642-8)
  
  __Util and evaluation classes__.
  
## To run:
  Follow test.py file
 
## TODO:
  - Fully document code
  - Improve Cython implementation
  - Add more algorithms
  - Provide a better usage manual