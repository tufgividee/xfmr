#!/bin/bash

uv run python -u bleu.py 2>&1 | tee bleu.log