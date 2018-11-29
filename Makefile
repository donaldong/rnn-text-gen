.PHONY: unittest 

unittest:
	python -m unittest discover -s ./test -p "*_test.py"
