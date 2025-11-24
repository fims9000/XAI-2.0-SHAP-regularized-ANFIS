"""
Post-hoc SHAP анализ
"""
import shap
import time
import numpy as np

class PostHocSHAPAnalyzer:
    """Класс для post-hoc SHAP анализа"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.task_type = config['dataset']['task_type']

    def analyze(self, X_test, feature_names):
        """Полный post-hoc SHAP анализ"""
        task_name = "регрессии" if self.task_type == 'regression' else "классификации"
        print(f"[INFO] Post-hoc SHAP анализ ({task_name})...")
        start_time = time.time()

        # Функция предсказания для SHAP
        if self.task_type == 'regression':
            def predict_fn(data):
                if hasattr(self.model, 'network'):
                    self.model.network.eval()
                return np.asarray(self.model.predict(data)).ravel()
        else:
            def predict_fn(data):
                if hasattr(self.model, 'predict_proba'):
                    preds = self.model.predict_proba(data)
                    # Для бинарной классификации берем вероятности класса 1
                    if len(preds.shape) > 1 and preds.shape[1] > 1:
                        return np.array(preds[:, 1]).ravel()
                    else:
                        return np.array(preds).ravel()
                else:
                    # Fallback: используем predict и преобразуем в вероятности
                    pred = self.model.predict(data)
                    return np.array(pred).ravel()

        # Создание explainer
        sample_size = self.config['shap']['sample_size']
        background = shap.sample(X_test, sample_size)
        explainer = shap.KernelExplainer(predict_fn, background)

        # Вычисление SHAP значений
        shap_values = explainer.shap_values(X_test[:sample_size])  # Ограничиваем размер
        analysis_time = time.time() - start_time

        # Глобальная важность
        global_importance = np.abs(shap_values).mean(axis=0)

        print(f"[OK] SHAP анализ завершен за {analysis_time:.2f} сек")

        results = {
            'explainer': explainer,
            'shap_values': shap_values,
            'global_importance': global_importance,
            'analysis_time': analysis_time,
            'feature_names': feature_names,
            'X_subset': X_test[:sample_size]  # Сохраняем подвыборку для визуализации
        }

        return results

    def create_shap_plots(self, results, save_dir=None):
        """Создание SHAP графиков"""
        shap_values = results['shap_values']
        X_subset = results['X_subset']
        feature_names = results['feature_names']

        # Summary plot
        shap.summary_plot(shap_values, X_subset, feature_names=feature_names, show=not bool(save_dir))

        # Bar plot
        shap.summary_plot(shap_values, X_subset, feature_names=feature_names,
                          plot_type="bar", show=not bool(save_dir))

        # Force plot для первого образца
        if len(shap_values) > 0:
            explainer = results['explainer']
            X_sample = np.round(X_subset[0], 2)
            shap.force_plot(explainer.expected_value, shap_values,
                            X_sample, feature_names=feature_names, matplotlib=True)
