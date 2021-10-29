from setuptools import setup, find_packages

import versioneer


with open('requirements.txt', mode='r', encoding='utf-8') as f:
    install_requires = f.read().splitlines()

with open("README.md") as readme:
    setup(name='previsionio',
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          description='python wrapper for prevision api',
          long_description=readme.read(),
          url='https://github.com/previsionio/prevision-python',
          author='prevision.io',
          author_email='team-core@prevision.io',
          packages=find_packages(exclude=['ci*', 'utest*']),
          license="MIT License",
          install_requires=install_requires,
          zip_safe=False,
          python_requires=">= 3.6",
          keywords='ml, ai, prevision, sdk',
          classifier=[
              'Intended Audience :: Developers',
              'Natural Language :: English',
              'License :: OSI Approved :: MIT License',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
              'Programming Language :: Python :: 3.8',
              'Programming Language :: Python :: 3.9',
          ]
          )
