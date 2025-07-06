# Arsenic Prediction Using Ensemble ML Techniques

## Repository Overview

This repository contains all the necessary code files, preprocessed datasets, trained models, and results for predicting arsenic levels in water using machine learning techniques. The project demonstrates the use of ensemble machine learning models to predict arsenic concentration based on various environmental and water quality features. 

## Project Structure

The repository is organized into the following directories and files:

### 1. **Dataset (original and preprocessed)**
   - Contains original and preprocessed datasets used for training and testing the machine learning models. The preprocessing includes data cleaning, normalization, and feature extraction.

### 2. **Saved Models**
   - Stores the trained ensemble models used for predictions. These models can be loaded and used directly to make predictions without retraining.

### 3. **finals**
   - Includes the final scripts and results related to the arsenic prediction project.
   - Files:
     - **Arsenic Prediction.ipynb**: Jupyter notebook that performs the complete process of training, evaluation, and prediction of arsenic levels using machine learning techniques.
     - **Copy of Arsenic Prediction.ipynb**: A copy of the main notebook for backup purposes.
     - **Updated_Arsenic_Prediction.ipynb**: A newer version of the prediction notebook with updates and improvements to the model.
     - **WaterQualityPrediction-v1.ipynb, WaterQualityPrediction-v2.ipynb, WaterQualityPrediction-v3.ipynb**: Sequential approach of the codes and models, contains all the main codes here
     - **Why Ensembled Model Performed Better.txt**: A text file discussing the performance improvements achieved by using ensemble models in comparison to individual machine learning models.
     - **classification_comparison_map.html, classification_comparison_map_full.html**: HTML files displaying the visual comparison of model performances in terms of classification accuracy and other metrics.

## Key Files and Notebooks

### Arsenic Prediction.ipynb
   - This notebook is the main code for training the machine learning models, making predictions, and evaluating the results. It uses ensemble techniques to improve the accuracy of predictions by combining multiple models.

### WaterQualityPrediction-v1.ipynb, WaterQualityPrediction-v2.ipynb, WaterQualityPrediction-v3.ipynb
   - These notebooks contains all the main codes of this project. v1 file contains the preprocessing steps, v2 file contains the model training and ensemble approach technicques, and finally v3 file contains the results and visualization techniques.
   - 

### Why Ensembled Model Performed Better.txt
   - This document explains the rationale behind using ensemble learning and its effectiveness in improving model performance.

### classification_comparison_map.html & classification_comparison_map_full.html
   - These files contain visual comparisons of the performance metrics of different models, allowing users to evaluate how each model performed in predicting arsenic levels.

## Requirements

To run the code, make sure to install the following libraries:

- Python 3.6 or later
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `tensorflow` or `keras` (depending on model type)
- `seaborn`
- `joblib` (for saving and loading models)

You can install the necessary libraries using `pip`:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost lightgbm tensorflow seaborn joblib
```
## How to Use

1. **Clone the repository**:

   Clone this repository to your local machine using the following command:

   ```bash
   git clone https://github.com/Hafiz-sustswe/Arsenic-Prediction-An-Ensemble-ML-Tecnique.git
   ```

2. **Prepare the dataset**:

   The dataset is already preprocessed and available in the `Dataset (original and preprocessed)` directory. If you want to start with a fresh dataset, you can load your raw data into the `Arsenic Prediction.ipynb` notebook and preprocess it accordingly.

3. **Train and evaluate the models**:

   * Open the notebook `Arsenic Prediction.ipynb` and run the cells sequentially to train the models and generate predictions.
   * You can experiment with different machine learning algorithms, adjust hyperparameters, or use the models in `WaterQualityPrediction-v1.ipynb` through `v3.ipynb`.

4. **Save and load models**:

   After training the models, they can be saved using `joblib` to store them for future use. Use the following code to save a model:

   ```python
   import joblib
   joblib.dump(model, 'path_to_model.pkl')
   ```

   To load the saved model, use:

   ```python
   model = joblib.load('path_to_model.pkl')
   ```

## Conclusion

This repository serves as a comprehensive implementation of ensemble machine learning models for arsenic prediction. The approach leverages various models to achieve better performance in predicting arsenic concentration in water, which is crucial for public health.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


