from os import path

from setuptools import setup, find_packages

# setup metainfo
libinfo_py = path.join('summ_eval', 'server', '__init__.py')
libinfo_content = open(libinfo_py, 'r').readlines()
version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][0]
exec(version_line)  # produce __version__

setup(
    name='summ_eval_server',
    version=__version__,
    description='Evaluating summaries with their corresponding reference summaries (Server)',
    url='https://github.com/andyweizhao/Evaluation_Metrics/',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    author='ABC',
    author_email='abc@gmail.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'numpy',
        'six',
        'pyzmq>=17.1.0',
        'termcolor>=1.1',
        'torch'
    ],
    extras_require={
        'http': ['flask', 'flask-compress', 'flask-cors', 'flask-json', 'summ-eval-client']
    },
    classifiers=(
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ),
    entry_points={
        'console_scripts': ['summ-eval-start=summ_eval.server.cli:main'],
    },
    keywords='summary evaluation pyramid rouge humman correlation',
)
