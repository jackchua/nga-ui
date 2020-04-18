#!/usr/bin/env bash

gunicorn run:app --config=./configs/gunicorn.py --reload