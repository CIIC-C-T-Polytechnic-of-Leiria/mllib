from setuptools import find_packages, setup
setup(
    name='mlciic',
    packages=find_packages(),
    version='0.2.0',
    description='Library to help with machine learning projects at CIIC C&T',
    author='CIIC C&T',
    license='MIT',
)
"""
setup(
    name='mypythonlib',
    packages=find_packages(include=['mypythonlib']),
    version='0.1.0',
    description='My first Python library',
    author='Me',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
"""