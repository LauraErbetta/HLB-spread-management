from setuptools import setup, find_packages
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np
import sys
import os

inc_path = [np.get_include()]
lib_path = [os.path.join(np.get_include(), '..', '..', 'random', 'lib')]

# Macros (define_macros=)
defs1 = [('NPY_NO_DEPRECATED_API', 0)]

# Detect platform
is_windows = sys.platform.startswith('win')
is_linux = sys.platform.startswith('linux')

if is_windows:
    optim = ['/O2']
    build_base = "hlb_model/build_win"
elif is_linux:
    optim = ['-O2']
    build_base = "hlb_model/build_linux"
else:
    raise RuntimeError("Unsupported platform")

c_output_dir = os.path.join(build_base, "c")

os.makedirs(c_output_dir, exist_ok=True)

# directives = {
#     "language_level": "3",
#     "boundscheck": False,
#     "wraparound": False,
#     "cdivision": True,
# }

directives = {"language_level": "3",
                "linetrace": False,
                "binding": False,
                "profile": False,
                "boundscheck": False,
                "wraparound": False,
                "nonecheck": False,
                "initializedcheck": False,
                "cdivision": True,}

extensions = [
    Extension("hlb_model.cy_scripts.extending_distributions", ["hlb_model/cy_scripts/extending_distributions.pyx"],
                          include_dirs=inc_path,
                          library_dirs=lib_path,
                          libraries=['npyrandom'],
                          define_macros = defs1,
                          extra_compile_args=optim),
    Extension("hlb_model.cy_scripts.ratesStructs", ["hlb_model/cy_scripts/ratesStructs.pyx"], include_dirs=inc_path, define_macros = defs1,
                          extra_compile_args=optim),
    Extension("hlb_model.cy_scripts.structs", ["hlb_model/cy_scripts/structs.pyx"], include_dirs=inc_path, define_macros = defs1,
                          extra_compile_args=optim),
    Extension("hlb_model.cy_scripts.linkedList", ["hlb_model/cy_scripts/linkedList.pyx"], include_dirs=inc_path, define_macros = defs1,
                          extra_compile_args=optim),
    Extension("hlb_model.cy_scripts.dispersal_updates", ["hlb_model/cy_scripts/dispersal_updates.pyx"], include_dirs=inc_path, define_macros = defs1,
                          extra_compile_args=optim),
    Extension("hlb_model.cy_scripts.management", ["hlb_model/cy_scripts/management.pyx"], include_dirs=inc_path, define_macros = defs1,
                          extra_compile_args=optim),
    Extension("hlb_model.cy_scripts.epidemic", ["hlb_model/cy_scripts/epidemic.pyx"], include_dirs=inc_path, define_macros = defs1,
                          extra_compile_args=optim),
    # Extension("hlb_model.cy_scripts.test_dispersal", ["hlb_model/cy_scripts/test_dispersal.pyx"], include_dirs=inc_path, define_macros = defs1,
    #                       extra_compile_args=optim),
]

setup(
    name="hlb_model",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "hlb_model": ["Data/**/*", "Data/*"]},
    ext_modules=cythonize(extensions,
                          build_dir = c_output_dir,
                          compiler_directives=directives
                        #   include_path=["hlb_model/cy_scripts"]
                          ),
)



