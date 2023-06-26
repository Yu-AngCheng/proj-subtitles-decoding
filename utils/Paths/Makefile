#!/bin/bash

all: test

test:
	python Paths.py
	cd ./eeg && make test

clean:
	rm -rf ./__pycache__
	cd ./eeg && make clean

