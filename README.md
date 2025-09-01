# Dog Detection

Dog Detection is a Python project powered by TensorFlow for detecting dogs in images.  
This repository provides scripts and models for building, training, and running dog detection using deep learning.

## Features

- Detects dogs in images using TensorFlow.
- Easy to train and deploy.
- Modular codebase for experimentation.

## Setup

1. **Clone the repository**
    ```bash
    git clone https://github.com/Braveheart66/dog-detection.git
    cd dog-detection
    ```
2. **Create a virtual environment (recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use venv\Scripts\activate
    ```
3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

- Add your training and test images to the appropriate directories.
- Train your model:
    ```bash
    python train.py
    ```
- Run dog detection on new images:
    ```bash
    python detect.py --image path/to/image.jpg
    ```

## Testing

- Run tests (if you have a `tests/` folder and use `pytest`):
    ```bash
    pytest
    ```

## Contributing

Contributions are welcome! Please open issues and pull requests to discuss improvements.

## License

MIT License

Copyright (c) 2025 Braveheart66

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
