.PHONY: unittest 

unittest:
	python -m unittest \
		test/alice_test.py \
		test/dataset_test.py \
		test/model_selector_test.py \
		test/text_generator_test.py
