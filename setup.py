import sys
from setuptools import setup, find_packages

python_major = sys.version_info[0]

install_requires = ['pandas',
                    'requests',
                    'apscheduler',
                    'seaborn',
                    'scikit-learn',
                    ]

if python_major == 2:
    install_requires.append('enum')

setup(name='previsionio',
      version='10.12.0',
      description='python wrapper for prevision api',
      url='',
      author='prevision.io',
      author_email='prevision@prevision.io',
      license='',
      packages=find_packages(),
      install_requires=install_requires,
      zip_safe=False,
      )
