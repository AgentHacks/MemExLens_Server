# MemExLens Server

A Python Flask API server that accepts JSON objects with timestamp, data, and url fields.

## Features

- **POST /api/data**: Accept JSON data with required fields (timestamp, data, url)
- **GET /api/data/<id>**: Retrieve specific data by ID
- **GET /health**: Health check endpoint
- Input validation for all required fields
- Error handling and logging
- JSON response format

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```