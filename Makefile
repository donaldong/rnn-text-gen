.PHONY: unittest 

unittest:
	python -m unittest test/dataset_test.py
	python -m unittest test/text_generator_test.py
