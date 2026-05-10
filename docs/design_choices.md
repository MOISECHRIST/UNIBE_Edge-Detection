# Design Choices & Architectural Decisions

This document outlines the technical decisions and architectural patterns used in the Edge-Detection Learning Platform.

## Architectural Overview

The application follows a modular architecture to ensure separation of concerns between the user interface, core processing logic, and visualization.

### 1. Separation of UI and Logic
- **`pages/`**: Contains the Streamlit-specific UI logic. Each page acts as a controller that handles user input (sliders, buttons, image selection) and calls functions from the core modules.
- **`modules/processing.py`**: The "Engine" of the project. It contains pure Python/NumPy implementations of image processing algorithms, isolated from the UI. This allows for easier testing and potential reuse in other contexts.

### 2. Streamlit for Interactivity
Streamlit was chosen for its ability to quickly create interactive data dashboards. 
- **Session State**: Used in `pages/1_edge_intuition.py` to persist selected points for intensity profiling across reruns.
- **Components**: Utilized `streamlit_image_coordinates` to enable direct interaction with images, bridging the gap between static plots and user input.

## Algorithm Implementation Details

### Edge Detection Philosophy
The project implements edge detection from first principles (gradients) to advanced multi-step algorithms (Canny).

- **Custom Kernels**: Instead of relying solely on black-box library functions, kernels for Sobel, Prewitt, and Roberts are explicitly defined in `modules/processing.py`. This serves an educational purpose, allowing users to understand the underlying convolution matrices.
- **Canny Breakdown**: To maximize learning, the Canny detector is not just called as a single function. Each step (NMS, Hysteresis) is exposed as a separate function. 
    - **Non-Maximum Suppression (NMS)**: Implemented manually to demonstrate how gradient direction is used to thin edges.
    - **Hysteresis**: A visualization function was created specifically to show the difference between "strong" and "weak" edges before final connectivity analysis.

### Noise Handling
- **Derivative of Gaussian (DoG)**: Implemented as an alternative to "Smoothing then Gradient". This demonstrates the associative property of convolution, showing that $\frac{d}{dx}(G * I) = (\frac{d}{dx}G) * I$.

## Visualization Strategy

- **Matplotlib Integration**: While Streamlit has native charting, Matplotlib was used for image processing results because of its fine-grained control over colormaps (e.g., `RdBu_r` for gradients, `hsv` for orientation) and subplots.
- **Consistency**: The `modules/img_viz.py` module ensures that all comparison plots share the same aesthetic (title sizes, axis removal, figure scaling).

## Performance Considerations

- **Vectorization**: Where possible, NumPy vectorized operations (e.g., gradient magnitude calculation) are used to maintain responsiveness in the interactive UI.
- **Efficient Convolution**: Utilizes `scipy.ndimage.convolve` for optimized 2D signal processing.
