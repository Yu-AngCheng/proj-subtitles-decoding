#!/bin/bash

all: test

test:
	cd ./layers && make test
	python defossez2022decoding.py

clean:
	rm -rf ./__pycache__
	cd ./layers && make clean

