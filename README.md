# pystream

An MOA-based implementation for data stream classification in Python/Cython

## Includes:

  __Base learners__:
  - Very Fast Decision Tree (https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf)
  - Strict Very Fast Decision Tree (https://www.sciencedirect.com/science/article/pii/S0167865518305580)
  - OLBoost (to be upload on arxiv)

  __Ensembles__:
  - OzaBag/OzaBoost (https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf)
  - OAUE (https://www.sciencedirect.com/science/article/pii/S0020025513008554)
  - LevBag (https://core.ac.uk/download/pdf/129931682.pdf)
  - ARF (https://link.springer.com/article/10.1007/s10994-017-5642-8)

  __Util and evaluation classes__.

## To run:
  - pip install -r requirements.txt --user
  - python setup.py build_ext --inplace (builds Cython extensions and creates .so files)
  - python setup.py install --user
  - Follow test.py file

## TODO:
  - Fully document code
  - Improve Cython implementation
  - Add more algorithms
  - Provide a better usage manual

## Note
  - Needs cython to compile code when installing
