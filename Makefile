.PHONY: unittest 

unittest:
	python -m unittest test/dataset_test.py \
		test/text_generator_test.py \
		test/alice_test.py
