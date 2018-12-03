.PHONY: unittest 

unittest:
	python -m unittest -s ./test -p "dataset_test.py"
	python -m unittest -s ./test -p "text_generator_test.py"
