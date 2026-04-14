#!/bin/bash
# Start AlphaNet development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
