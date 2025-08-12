# XAI 2.0: SHAP Regularization for ANFIS Neural-Fuzzy Networks

Practical implementation of embedded SHAP regularization method in Adaptive Neuro-Fuzzy Inference System (ANFIS) models to enhance interpretability without sacrificing accuracy.

## 🎯 Key Features

- **Vanilla ANFIS**: Training standard neural-fuzzy models
- **SHAP Post-hoc Analysis**: Traditional feature importance analysis after training
- **SHAP-Regularized ANFIS**: Embedded regularization for enhanced interpretability
- **Universal Support**: Classification and regression tasks
- **Automated Visualization**: Comprehensive plots and metrics
- **Flexible Configuration**: YAML configurations for various datasets

## 📁 Project Structure

```
XAI-2.0-SHAP-regularized-ANFIS/
├── configs/                    # YAML dataset configurations
│   ├── breast_cancer.yaml
│   ├── heart_disease.yaml
│   ├── diabetes.yaml
│   ├── banknote_auth.yaml
│   └── wine_quality.yaml
├── src/
│   ├── data/
│   │   └── loader.py          # Universal data loader
│   ├── models/
│   │   ├── anfis_manager.py   # ANFIS model manager
│   │   └── shap_trainer.py    # SHAP-regularized trainer
│   ├── analysis/
│   │   └── shap_analyzer.py   # Post-hoc SHAP analysis
│   └── visualization/
│       └── visualizer.py      # Plots and metrics creation
├── experiments/
│   └── run_experiment.py      # Main execution script
├── results/                   # Experiment results
└── datasets/                  # Dataset folder
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Main Dependencies:**
- `xanfis` - Neural-fuzzy systems library
- `shap` - Feature importance analysis
- `scikit-learn` - Machine learning
- `matplotlib`, `seaborn` - Visualization
- `pandas`, `numpy` - Data processing

### 2. Dataset Preparation

Place your dataset in the `src/datasets/` folder or use ready-made configurations for popular datasets.

### 3. Running Experiments

**Full experiment (all methods):**
```bash
python experiments/run_experiment.py --dataset breast_cancer --experiment all --save-results
```

**Vanilla ANFIS only:**
```bash
python experiments/run_experiment.py --dataset heart_disease --experiment vanilla
```

**SHAP analysis only:**
```bash
python experiments/run_experiment.py --dataset diabetes --experiment shap --save-results
```

**SHAP regularization only:**
```bash
python experiments/run_experiment.py --dataset wine_quality --experiment regularized
```

## 📊 Supported Datasets

| Dataset | Config | Task | Description |
|---------|---------|---------|-----------|
| Wisconsin Breast Cancer | `breast_cancer` | Classification | Breast cancer diagnosis |
| Heart Disease | `heart_disease` | Classification | Heart disease prediction |
| Pima Indians Diabetes | `diabetes` | Classification | Diabetes diagnosis |
| Banknote Authentication | `banknote_auth` | Classification | Banknote authentication |
| Red Wine Quality | `wine_quality` | Regression | Wine quality assessment |

## ⚙️ Creating Custom Configuration

Create a file `configs/my_dataset.yaml`:

```yaml
# Configuration for your dataset
dataset:
  name: "My Dataset"
  source_type: "local"                    # "local" or "url"
  file_path: "./src/datasets/my_data.csv"
  description: "Description of your dataset"
  task_type: "classification"             # "classification" or "regression"

  special_preprocessing:
    type: "standard"                      # Processing type
    has_header: true                      # Whether headers exist
    target_column: "target"               # Target column name
    target_mapping: null                  # Value mapping (if needed)

preprocessing:
  test_size: 0.25                         # Test set size
  random_state: 42                        # Random seed
  scaling: true                           # Feature scaling
  stratify: true                          # Stratified split

model:
  num_rules: 5                            # Number of fuzzy rules
  mf_class: "GBell"                       # Membership function type
  optim: "OriginalPSO"                    # Optimization algorithm
  optim_params:
    epoch: 20                             # Number of epochs
    pop_size: 30                          # Population size
    verbose: true
  reg_lambda: 0.1                         # Regularization coefficient
  seed: 42

shap:
  gamma: 0.5                              # SHAP regularization coefficient
  training_epochs: 25                     # Epochs for SHAP training
  batch_size: 32                          # Batch size
  learning_rate: 0.005                    # Learning rate
  sample_size: 80                         # Sample size for SHAP

visualization:
  figure_size: [20, 16]                   # Plot size
  style: "seaborn-v0_8"                   # Plot style
  save_plots: true                        # Save plots
  plot_format: "png"                      # File format
  dpi: 300                                # Resolution
```

Run the experiment:
```bash
python experiments/run_experiment.py --dataset my_dataset --experiment all
```

## 📈 Experiment Results

After execution, the `results/` folder will contain:

- **Analysis plots**: ROC curves, feature importance, confusion matrices
- **Comparative metrics**: CSV table with results from all methods
- **SHAP visualizations**: Summary plots, waterfall plots, force plots
- **Execution log**: Detailed information about the training process

## 🔬 Research Methods

### 1. Vanilla ANFIS
- Standard neural-fuzzy system training
- Feature importance extraction from model coefficients
- Baseline accuracy for comparison

### 2. Post-hoc SHAP Analysis
- Traditional approach using the `shap` library
- KernelExplainer for model-agnostic analysis
- Local and global explanations

### 3. SHAP-Regularized ANFIS
- Embedded SHAP-like regularization in the loss function
- Simultaneous optimization of accuracy and interpretability
- Reduced computational costs

## 🎛️ Parameter Tuning

### ANFIS Model
- `num_rules`: Number of fuzzy rules (5-15 for most tasks)
- `mf_class`: Membership function type (`GBell`, `Gaussian`, `Sigmoid`)
- `optim`: Optimization algorithm (`OriginalPSO`, `BaseGA`, `OriginalABC`)

### SHAP Regularization
- `gamma`: SHAP regularization weight (0.1-1.0)
- `training_epochs`: Number of epochs (15-50)
- `learning_rate`: Learning rate (0.001-0.01)

## 📊 Evaluation Metrics

### Classification
- **Accuracy**: Overall accuracy
- **Precision**: Positive class precision
- **Recall**: Recall (sensitivity)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve

### Regression
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

## 👥 Authors

- **Yuri Trofimov**
- **Aleksey Shevchenko**
- **Andrei Ilin**
- **Alexander Lebedev**
- **Aleksey Averkin**

## 📚 Citation

If you use this code in your research, please cite our work:

### APA Style
```
Trofimov, Y., Shevchenko, A., Ilin, A., Lebedev, A., & Averkin, A. (2025). 
XAI 2.0: Embedded SHAP Regularization as a Principle for Building Globally 
and Locally Interpretable Models (Version v1.0.0) [Computer software]. 
Zenodo. https://doi.org/10.5281/zenodo.16790521
```

### BibTeX
```bibtex
@software{trofimov2025xai,
  title={XAI 2.0: Embedded SHAP Regularization as a Principle for Building Globally and Locally Interpretable Models},
  author={Trofimov, Yuri and Shevchenko, Aleksey and Ilin, Andrei and Lebedev, Alexander and Averkin, Aleksey},
  year={2025},
  version={v1.0.0},
  publisher={Zenodo},
  doi={10.5281/zenodo.16790521},
  url={https://doi.org/10.5281/zenodo.16790521},
  note={FUZZY\_XAI}
}
```

### IEEE Style
```
Y. Trofimov, A. Shevchenko, A. Ilin, A. Lebedev, and A. Averkin, 
"XAI 2.0: Embedded SHAP Regularization as a Principle for Building 
Globally and Locally Interpretable Models," Zenodo, v1.0.0, 2025. 
[Online]. Available: https://doi.org/10.5281/zenodo.16790521
```

## 📄 License

This project is part of a scientific research published on Zenodo with DOI: `10.5281/zenodo.16790521`.

**Usage Terms:**
- ✅ Academic and research use is welcomed
- ✅ Commercial use is permitted with proper attribution
- ✅ Modification and distribution with reference to the original work
- ⚠️ **Mandatory citation** when used in publications

### Disclaimer

The software is provided "as is", without any warranties, express or implied. The authors are not responsible for any losses or damages arising from the use of this software.

## 🏛️ Institutional Affiliation

This work was performed as part of research in explainable artificial intelligence (XAI 2.0) and fuzzy neural systems.

## 📖 Related Publications

The complete methodology and theoretical foundation are presented in the paper available at: https://doi.org/10.5281/zenodo.16790521

## 🆘 Support

If you encounter questions or issues:

1. Check [Issues](../../issues) for similar problems
2. Create a new Issue with detailed description
3. Ensure all dependencies are installed correctly

***

**This project is intended for researchers and practitioners in explainable AI interested in enhancing model interpretability without sacrificing accuracy.**