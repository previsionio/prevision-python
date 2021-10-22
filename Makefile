lint:
	pycodestyle --config pep8lint.cfg --statistics --format=pylint *.py previsionio/*.py utests/*.py
	pyflakes  *.py previsionio/*.py utests/*.py

utests:
	cd utests; pytest

utests-ci:
	cd utests; py.test --junitxml=test_results.xml --cov=../previsionio --cov-config .coveragerc --cov-report html; coverage report -m

# docker.bin:
#     cd ci; sh Makefile

# docker:

.PHONY: utests
