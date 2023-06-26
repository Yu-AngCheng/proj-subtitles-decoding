#!/bin/bash

all: test

test:
	cd ./eeg && make test
	cd ./meg && make test

clean:
	rm -rf ./__pycache__
	cd ./eeg && make clean
	cd ./meg && make clean

