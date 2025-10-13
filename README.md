# Disease Predictor Deep Learning

A deep learning-based disease prediction system using TensorFlow and machine learning techniques.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a disease prediction model using deep learning techniques. The system analyzes medical data to predict potential diseases and provides interpretable results through SHAP analysis.

## Features

- 🧠 Deep learning-based disease prediction
- 📊 Data visualization and analysis
- 🔍 Model interpretability with SHAP
- 🌐 Web interface using Flask
- 📈 Performance metrics and evaluation
- 📓 Jupyter notebooks for experimentation

## Project Structure

```
disease-predictor-deep-learning/
├── .qodo/              # Qodo AI tools configuration
├── data/               # Dataset files
├── notebooks/          # Jupyter notebooks for analysis
├── .gitignore         # Git ignore rules
├── LICENSE            # Project license
├── README.md          # Project documentation
└── requirements.txt   # Python dependencies
```

## Installation

### Prerequisites
- Python 3.10
- Miniconda or Anaconda
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/disease-predictor-deep-learning.git
cd disease-predictor-deep-learning
```

### Step 2: Create Conda Environment

```bash
# Create a new conda environment with Python 3.10
conda create -n disease-predictor python=3.10

# Activate the environment
conda activate disease-predictor
```

### Step 3: Install Dependencies

#### Option A: Install from requirements.txt (Recommended)
```bash
pip install -r requirements.txt
```

#### Option B: Manual Installation
```bash
# Core ML and DL packages
pip install tensorflow==2.16.1
pip install scikit-learn numpy pandas

# Visualization
pip install matplotlib seaborn plotly

# Image processing
pip install opencv-python pillow

# Web framework
pip install flask

# Model interpretability and utilities
pip install shap tqdm

# Development tools
pip install jupyterlab notebook
```

### Step 4: Export Environment (Optional)

Save your environment configuration for reproducibility:

```bash
# Export to YAML file
conda env export > environment.yml
```

To recreate the environment from the YAML file:

```bash
# Create environment from file
conda env create -f environment.yml
```

## Usage

### Running Jupyter Notebooks

```bash
# Activate environment
conda activate disease-predictor

# Start Jupyter Lab
jupyter lab

# Or start Jupyter Notebook
jupyter notebook
```

### Running the Flask Application

```bash
# Activate environment
conda activate disease-predictor

# Run the Flask app
python app.py
```

### Training the Model

```bash
# Activate environment
conda activate disease-predictor

# Run training script
python train.py
```

## Technologies Used

### Core Frameworks
- **TensorFlow 2.16.1** - Deep learning framework
- **Scikit-learn** - Machine learning utilities
- **Flask** - Web application framework

### Data Processing
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis

### Visualization
- **Matplotlib** - Static plotting
- **Seaborn** - Statistical visualization
- **Plotly** - Interactive visualizations

### Image Processing
- **OpenCV** - Computer vision tasks
- **Pillow** - Image processing

### Model Interpretability
- **SHAP** - Model explanation and interpretability

### Development Tools
- **JupyterLab** - Interactive development environment
- **tqdm** - Progress bars

## Environment Management

### Verify Active Environment
```bash
conda info --envs
```

### Deactivate Environment
```bash
conda deactivate
```

### Delete Environment (if needed)
```bash
conda env remove -n disease-predictor
```

### List Installed Packages
```bash
conda list
# or
pip list
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the terms specified in the LICENSE file.

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: Make sure to activate the conda environment before running any scripts or notebooks:
```bash
conda activate disease-predictor
```