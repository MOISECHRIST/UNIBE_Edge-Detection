# Edge Detection Learning Platform

An interactive, educational application designed to explore the mathematical and practical foundations of edge detection in image analysis. Built with Streamlit, this platform provides a hands-on approach to understanding how computers "see" boundaries and structures within digital images.

## Features

- **Edge Intuition:** Interactive visualization of pixel intensity profiles. Click two points on an image to see the 1D intensity signal and its derivative.
- **Gradient-Based Detection:** Explore first-order derivative filters including **Sobel**, **Prewitt**, and **Roberts** operators.
- **Noise Robustness:** Analyze the impact of noise on edge detection and compare solutions like Gaussian smoothing and Derivative of Gaussian (DoG) kernels.
- **Canny Deep Dive:** A step-by-step breakdown of the Canny Edge Detector:
    1. Gaussian Smoothing
    2. Gradient Calculation
    3. Non-Maximum Suppression (NMS)
    4. Hysteresis Thresholding
- **Laplacian Methods:** Compare second-order derivative techniques using the Laplacian and Laplacian of Gaussian (LoG) filters.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MOISECHRIST/UNIBE_Edge-Detection.git 
   cd UNIBE_Edge-Detection
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To launch the interactive application:
```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main entry point and navigation setup.
- `pages/`: Individual Streamlit pages for each learning module.
- `modules/`:
    - `processing.py`: Core image processing algorithms and filter implementations.
    - `utils.py`: Image I/O, grayscale conversion, and noise utilities.
    - `img_viz.py`: Matplotlib-based visualization wrappers.
- `sample_data/`: A collection of standard test images.

## Author
**MEKA Moïse Christian Junior**  
Email: [moise.meka@students.unibe.ch](mailto:moise.meka@students.unibe.ch)
