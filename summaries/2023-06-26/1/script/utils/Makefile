#!/bin/bash

all: test

test:
	python DotDict.py
	python model.py
	cd ./Paths && make test
	cd ./data && make test

clean:
	rm -rf ./__pycache__
	cd ./Paths && make clean
	cd ./data && make clean

