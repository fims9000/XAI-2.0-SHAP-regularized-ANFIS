# XAI 2.0: SHAP Regularization for ANFIS Neural-Fuzzy Networks

Practical implementation of embedded SHAP regularization method in Adaptive Neuro-Fuzzy Inference System (ANFIS) models to enhance interpretability without sacrificing accuracy.

## Key Features

The system provides three main approaches: Vanilla ANFIS for training standard neural-fuzzy models, SHAP Post-hoc Analysis for traditional feature importance analysis after training, and SHAP-Regularized ANFIS with embedded regularization for enhanced interpretability. The system supports both classification and regression tasks, provides automated visualization with comprehensive plots and metrics, and uses flexible YAML configurations for various datasets.

## Project Structure

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

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include xanfis for neural-fuzzy systems, shap for feature importance analysis, scikit-learn for machine learning, matplotlib and seaborn for visualization, and pandas with numpy for data processing.

### Dataset Preparation

Place your dataset in the `src/datasets/` folder or use ready-made configurations for popular datasets.

### Running Experiments

For a full experiment with all methods:

```bash
python experiments/run_experiment.py --dataset breast_cancer --experiment all --save-results
```

To run only Vanilla ANFIS:

```bash
python experiments/run_experiment.py --dataset heart_disease --experiment vanilla
```

For SHAP analysis only:

```bash
python experiments/run_experiment.py --dataset diabetes --experiment shap --save-results
```

For SHAP regularization only:

```bash
python experiments/run_experiment.py --dataset wine_quality --experiment regularized
```

## Supported Datasets

The system supports several datasets: Wisconsin Breast Cancer for breast cancer diagnosis, Heart Disease for heart disease prediction, Pima Indians Diabetes for diabetes diagnosis, Banknote Authentication for banknote authentication, and Red Wine Quality for wine quality assessment.

## Creating Custom Configuration

Create a file `configs/my_dataset.yaml` with your dataset configuration. The configuration includes dataset information such as name, source type, file path, description, and task type. Preprocessing parameters include test size, random state, scaling, and stratification. Model parameters include number of fuzzy rules, membership function type, optimization algorithm, and regularization coefficient. SHAP parameters include gamma coefficient, training epochs, batch size, learning rate, and sample size. Visualization parameters include figure size, style, save options, format, and DPI.

Run the experiment with:

```bash
python experiments/run_experiment.py --dataset my_dataset --experiment all
```

## Experiment Results

After execution, the `results/` folder will contain analysis plots including ROC curves, feature importance, and confusion matrices. Comparative metrics are provided in CSV format with results from all methods. SHAP visualizations include summary plots, waterfall plots, and force plots. Execution logs contain detailed information about the training process.

### Key Results (Breast Cancer Dataset)

The system achieved ROC-AUC improvement from 99.43% to 99.85%, representing a 0.42% increase. The method is 3.6 times faster than Vanilla + Post-hoc SHAP approach, completing in 5 seconds versus 18 seconds. Interpretability is built-in with explanations always consistent with model logic. All visualizations include 12 high-quality plots ready for presentation.

For complete system documentation, see `results/breast_cancer/SYSTEM_DOCUMENTATION.md`

## Research Methods

### Vanilla ANFIS

Standard neural-fuzzy system training with feature importance extraction from model coefficients, providing baseline accuracy for comparison.

### Post-hoc SHAP Analysis

Traditional approach using the shap library with KernelExplainer for model-agnostic analysis, providing both local and global explanations.

### SHAP-Regularized ANFIS

Embedded SHAP-like regularization in the loss function enables simultaneous optimization of accuracy and interpretability while reducing computational costs.

## Parameter Tuning

For ANFIS model, the number of fuzzy rules typically ranges from 5 to 15 for most tasks. Membership function types include GBell, Gaussian, and Sigmoid. Optimization algorithms include OriginalPSO, BaseGA, and OriginalABC.

For SHAP regularization, the gamma coefficient ranges from 0.1 to 1.0, training epochs from 15 to 50, and learning rate from 0.001 to 0.01.

## Evaluation Metrics

For classification tasks, the system evaluates accuracy, precision, recall, F1-Score, and ROC AUC. For regression tasks, it evaluates RMSE, MAE, and R² coefficient of determination.

## Contributing

To contribute, fork the repository, create a feature branch, commit your changes, push to the branch, and create a Pull Request.

## Authors

- Yuri Trofimov
- Aleksey Shevchenko
- Andrei Ilin
- Alexander Lebedev
- Aleksey Averkin

## Citation

If you use this code in your research, please cite our work using one of the following formats:

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
[Online]. Available: https://doi.org/10.5281/zenodo.16790520
```

## License

This project is part of a scientific research published on Zenodo with DOI: `10.5281/zenodo.16790521`.

Usage Terms: Academic and research use is welcomed. Commercial use is permitted with proper attribution. Modification and distribution with reference to the original work is allowed. Mandatory citation is required when used in publications.

### Disclaimer

The software is provided "as is", without any warranties, express or implied. The authors are not responsible for any losses or damages arising from the use of this software.

## Institutional Affiliation

The work was performed within the framework of the state assignment of the Ministry of Science and Higher Education of the Russian Federation (topic No. 124112200072-2).

## Related Publications

The complete methodology and theoretical foundation are presented in the paper available at: https://doi.org/10.5281/zenodo.16790521

## Support

If you encounter questions or issues, check Issues for similar problems, create a new Issue with detailed description, or ensure all dependencies are installed correctly.

---

This project is intended for researchers and practitioners in explainable AI interested in enhancing model interpretability without sacrificing accuracy.
