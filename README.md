# disease-predictor-deep-learning




conda create -n disease-predictor python=3.10
conda activate disease-predictor


pip install tensorflow scikit-learn numpy pandas matplotlib seaborn opencv-python pillow
pip install flask


pip install shap tqdm plotly
pip install jupyterlab notebook
pip install tensorflow==2.16.1



conda env export > environment.yml
conda env create -f environment.yml
