#!/bin/bash

all: test

test:
	python lasso_regression.py
	python logistic_regression.py
	python naive_cnn.py
	cd ./defossez2022decoding && make test

clean:
	rm -rf ./__pycache__
	cd ./defossez2022decoding && make clean

