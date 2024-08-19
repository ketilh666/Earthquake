from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='earthquake',
    version='1.0',
    description='Functions for earthquae data analysis',
    author='KetilH',
    author_email='kehok@equinor.com',
    packages=['earthquake'],  #same as name
    install_requires=[], # avoid reinstalling stuff
    # install_requires=required, #external packages as dependencies
    zip_safe=False,
)