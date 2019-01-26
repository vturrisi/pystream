from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import Cython
import numpy as np
import os

libs = []
if os.name == 'posix':
    libs.append('m')

dirs = [np.get_include(), '.']

extensions = []

# ----------------------------- algorithms -----------------------------

# ----------------------------- base -----------------------------

dependencies = ['pystream/algorithms/nominal_counters/nominal_counter.pxd'
                'pystream/algorithms/nominal_counters/nominal_counter.pxd']

E = Extension(name='pystream.algorithms.base.vfdt',
              sources=['pystream/algorithms/base/vfdt.pyx'],
              libraries=libs,
              include_dirs=dirs,
              depends=dependencies)
extensions.append(E)

E = Extension(name='pystream.algorithms.base.svfdt',
              sources=['pystream/algorithms/base/svfdt.pyx'],
              libraries=libs,
              include_dirs=dirs,
              depends=dependencies + ['pystream/algorithms/base/vfdt.pxd'])
extensions.append(E)

E = Extension(name='pystream.algorithms.base.svfdt_ii',
              sources=['pystream/algorithms/base/svfdt_ii.pyx'],
              libraries=libs,
              include_dirs=dirs,
              depends=dependencies + ['pystream/algorithms/base/vfdt.pxd',
                                      'pystream/algorithms/base/svfdt.pxd'])
extensions.append(E)

# ----------------------------- change_detectors -----------------------------

E = Extension(name='pystream.algorithms.change_detectors.adwin',
              sources=['pystream/algorithms/change_detectors/adwin.pyx'],
              libraries=libs,
              include_dirs=dirs)
extensions.append(E)

# ----------------------------- ensembles -----------------------------

dependencies = ['pystream/algorithms/change_detectors/adwin.pxd']

E = Extension(name='pystream.algorithms.ensembles.arf',
              sources=['pystream/algorithms/ensembles/arf.pyx'],
              libraries=libs,
              include_dirs=dirs,
              depends=dependencies)
extensions.append(E)

E = Extension(name='pystream.algorithms.ensembles.leveraging_bagging',
              sources=['pystream/algorithms/ensembles/leveraging_bagging.pyx'],
              libraries=libs,
              include_dirs=dirs,
              depends=dependencies)
extensions.append(E)

E = Extension(name='pystream.algorithms.ensembles.oaue',
              sources=['pystream/algorithms/ensembles/oaue.pyx'],
              libraries=libs,
              include_dirs=dirs)
extensions.append(E)

E = Extension(name='pystream.algorithms.ensembles.ozabagging',
              sources=['pystream/algorithms/ensembles/ozabagging.pyx'],
              libraries=libs,
              include_dirs=dirs)
extensions.append(E)

E = Extension(name='pystream.algorithms.ensembles.ozaboosting',
              sources=['pystream/algorithms/ensembles/ozaboosting.pyx'],
              libraries=libs,
              include_dirs=dirs)
extensions.append(E)

# ----------------------------- nominal_counters -----------------------------

E = Extension(name='pystream.algorithms.base.nominal_counters.nominal_counter',
              sources=['pystream/algorithms/base/nominal_counters/nominal_counter.pyx'],
              libraries=libs,
              include_dirs=dirs)
extensions.append(E)

# ----------------------------- numeric_estimators -----------------------------

E = Extension(name='pystream.algorithms.base.numeric_estimators.gaussian_estimator',
              sources=['pystream/algorithms/base/numeric_estimators/gaussian_estimator.pyx'],
              libraries=libs,
              include_dirs=dirs)
extensions.append(E)

# ----------------------------- statistics -----------------------------

E = Extension(name='pystream.algorithms.base.statistics.tree_stats',
              sources=['pystream/algorithms/base/statistics/tree_stats.pyx'],
              libraries=libs,
              include_dirs=dirs)
extensions.append(E)

E = Extension(name='pystream.algorithms.base.statistics.value_stats',
              sources=['pystream/algorithms/base/statistics/value_stats.pyx'],
              libraries=libs,
              include_dirs=dirs)
extensions.append(E)


# ----------------------------- evaluation -----------------------------

E = Extension(name='pystream.evaluation.evaluate_prequential',
              sources=['pystream/evaluation/evaluate_prequential.pyx'],
              libraries=libs,
              include_dirs=dirs)
extensions.append(E)

E = Extension(name='pystream.evaluation.performance_statistics',
              sources=['pystream/evaluation/performance_statistics.pyx'],
              libraries=libs,
              include_dirs=dirs)
extensions.append(E)

# ----------------------------- utils -----------------------------

E = Extension(name='pystream.utils.stream_gen',
              sources=['pystream/utils/stream_gen.pyx'],
              libraries=libs,
              include_dirs=dirs)
extensions.append(E)

setup(name='pystream',
      version='1.0',
      description='Pystream',
      url='http://github.com/vturrisi/pystream',
      author='Victor Turrisi',
      license='MIT',
      packages=find_packages(exclude=['tests']),
      quiet=True,
      ignore_setup_xxx_py=True,
      assume_default_configuration=True,
      ext_modules=cythonize(extensions, nthreads=4, quiet=False),
      cmdclass={'build_ext': Cython.Build.build_ext},
      setup_requires=['cython>=0.x'],
      install_requires=['numpy>=1.14.1', 'pandas>=0.20.0', 'cython>=0.x'],
      include_package_data=True,
      zip_safe=False)