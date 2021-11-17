lint:
	pycodestyle --config pep8lint.cfg --statistics --format=pylint *.py previsionio/*.py utests/*.py
	pyflakes . previsionio/ utests/

utests:
	cd utests; py.test

utests-ci:
	cd utests; py.test --junitxml=test_results.xml --cov=../previsionio --cov-config .coveragerc --cov-report html; coverage report -m

.PHONY: utests
