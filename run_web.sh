#!/usr/bin/env bash

gunicorn run:app --config=./configs/gunicorn.py --bind 0.0.0.0:5000 --reload
