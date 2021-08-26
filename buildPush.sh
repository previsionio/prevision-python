rm dist/*
python setup.py sdist bdist_wheel
twine upload dist/*
rm -rf build/lib/*
rm -rf dist/*