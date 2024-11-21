#!/bin/bash

poetry run uvicorn api:app --port=80 --host="0.0.0.0" --log-level=debug --lifespan=on