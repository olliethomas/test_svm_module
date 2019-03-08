from distutils.core import setup

setup(
    name='testsvm',
    version='0.1.0',
    author='O. Thomas',
    packages=['exsvm'],
    description='Framework for algorithmic fairness comparisons',
    python_requires="==2.7",
    install_requires=[
        "pygame",
    ],
)
