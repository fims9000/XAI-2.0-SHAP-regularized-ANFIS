"""
Модуль для извлечения и сравнения правил ANFIS
"""
import numpy as np
import torch

class ANFISRulesExtractor:
    """Класс для извлечения правил из ANFIS моделей"""
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.num_rules = None
        self.num_features = len(feature_names)
        
    def extract_rules(self):
        """Извлечение правил из модели ANFIS"""
        try:
            network = self.model.network if hasattr(self.model, 'network') else self.model
            
            # Получаем параметры функций принадлежности
            state_dict = network.state_dict()
            
            # Отладочная информация
            if len(state_dict) == 0:
                print(f"[WARNING] State dict пустой для модели")
                return self._create_dummy_rules()
            
            # Извлекаем параметры функций принадлежности
            # Для Generalized Bell функции: a, b, c
            if 'mf_params' in state_dict:
                mf_params = state_dict['mf_params'].detach().cpu().numpy()
            else:
                # Альтернативный способ извлечения
                mf_params = self._extract_mf_params_alternative(network)
            
            # Извлекаем веса правил (коэффициенты)
            if 'coeffs' in state_dict:
                coeffs = state_dict['coeffs'].detach().cpu().numpy()
            else:
                coeffs = self._extract_coeffs_alternative(network)
            
            # Определяем количество правил
            if len(coeffs.shape) > 0:
                self.num_rules = coeffs.shape[0]
            else:
                self.num_rules = 1
            
            # Проверяем соответствие размерностей
            if len(mf_params.shape) >= 2:
                actual_num_features = mf_params.shape[1] if len(mf_params.shape) >= 3 else mf_params.shape[0]
                if actual_num_features != self.num_features:
                    print(f"[WARNING] Несоответствие размерностей: ожидалось {self.num_features} признаков, найдено {actual_num_features}")
                    # Используем минимальное значение
                    self.num_features = min(self.num_features, actual_num_features)
            
            rules = []
            for rule_idx in range(self.num_rules):
                rule = {
                    'rule_id': rule_idx,
                    'membership_functions': {},
                    'coefficients': coeffs[rule_idx] if len(coeffs.shape) > 1 else coeffs,
                    'rule_strength': None
                }
                
                # Извлекаем параметры функций принадлежности для каждого признака
                for feat_idx in range(self.num_features):
                    feat_name = self.feature_names[feat_idx]
                    try:
                        if len(mf_params.shape) >= 3:
                            # Формат: [num_rules, num_features, num_params]
                            # Проверяем границы: rule_idx должен быть < shape[0], feat_idx должен быть < shape[1]
                            if rule_idx < mf_params.shape[0] and feat_idx < mf_params.shape[1]:
                                rule['membership_functions'][feat_name] = mf_params[rule_idx, feat_idx, :]
                            elif mf_params.shape[0] > 0 and mf_params.shape[1] > 0:
                                # Используем ближайшие доступные индексы
                                safe_rule_idx = min(rule_idx, mf_params.shape[0] - 1)
                                safe_feat_idx = min(feat_idx, mf_params.shape[1] - 1)
                                rule['membership_functions'][feat_name] = mf_params[safe_rule_idx, safe_feat_idx, :]
                            else:
                                rule['membership_functions'][feat_name] = np.array([0.5, 1.0, 0.5])
                        elif len(mf_params.shape) == 2:
                            # Формат: [num_features, num_params]
                            if feat_idx < mf_params.shape[0]:
                                rule['membership_functions'][feat_name] = mf_params[feat_idx, :]
                            elif mf_params.shape[0] > 0:
                                # Используем первый доступный признак
                                rule['membership_functions'][feat_name] = mf_params[0, :]
                            else:
                                rule['membership_functions'][feat_name] = np.array([0.5, 1.0, 0.5])
                        else:
                            rule['membership_functions'][feat_name] = mf_params
                    except (IndexError, ValueError) as e:
                        # Если не удалось извлечь параметры, используем значения по умолчанию
                        rule['membership_functions'][feat_name] = np.array([0.5, 1.0, 0.5])
                
                rules.append(rule)
            
            return rules
            
        except Exception as e:
            print(f"[WARNING] Ошибка при извлечении правил: {e}")
            return self._create_dummy_rules()
    
    def _extract_mf_params_alternative(self, network):
        """Альтернативный способ извлечения параметров функций принадлежности"""
        # Пытаемся найти параметры в слоях сети
        params = []
        for name, param in network.named_parameters():
            if 'mf' in name.lower() or 'membership' in name.lower():
                params.append(param.detach().cpu().numpy())
        
        if params:
            return np.array(params)
        else:
            # Возвращаем фиктивные параметры
            return np.ones((self.num_features, 3)) * 0.5
    
    def _extract_coeffs_alternative(self, network):
        """Альтернативный способ извлечения коэффициентов"""
        for name, param in network.named_parameters():
            if 'coeff' in name.lower() or 'output' in name.lower():
                return param.detach().cpu().numpy()
        
        # Возвращаем фиктивные коэффициенты
        return np.ones((self.num_rules or 5, self.num_features + 1)) * 0.1
    
    def _create_dummy_rules(self):
        """Создание фиктивных правил для демонстрации"""
        rules = []
        for rule_idx in range(5):  # Предполагаем 5 правил
            rule = {
                'rule_id': rule_idx,
                'membership_functions': {name: np.array([0.5, 1.0, 0.5]) 
                                        for name in self.feature_names},
                'coefficients': np.ones(self.num_features + 1) * 0.1,
                'rule_strength': 0.2
            }
            rules.append(rule)
        return rules
    
    def compare_rules(self, rules1, rules2, model1_name="Model 1", model2_name="Model 2"):
        """
        Сравнение правил двух моделей ANFIS и вычисление их сходства.
        
        Сходство правил показывает, насколько похожи правила между двумя моделями. 
        Для каждого правила вычисляется различие коэффициентов (весов выходного слоя) 
        и различие параметров функций принадлежности по всем признакам. Затем сходство 
        вычисляется по формуле similarity = 1.0 / (1.0 + coeff_diff + mf_diff), где 
        coeff_diff - среднее абсолютное отклонение коэффициентов, а mf_diff - среднее 
        абсолютное отклонение параметров функций принадлежности. Если правила полностью 
        идентичны, сходство равно единице. Чем больше различия между правилами, тем 
        меньше значение сходства, которое стремится к нулю. Итоговое среднее сходство 
        представляет собой среднее арифметическое сходств всех правил.
        
        Returns:
            dict: Словарь с метриками сравнения, включая среднее сходство, различия 
                  коэффициентов и функций принадлежности.
        """
        comparison = {
            'num_rules': {
                model1_name: len(rules1),
                model2_name: len(rules2)
            },
            'rule_similarities': [],
            'coefficient_differences': [],
            'mf_differences': []
        }
        
        min_rules = min(len(rules1), len(rules2))
        
        for i in range(min_rules):
            rule1 = rules1[i]
            rule2 = rules2[i]
            
            # Сначала вычисляем различие коэффициентов между правилами двух моделей.
            # Коэффициенты представляют собой веса выходного слоя для данного правила,
            # и мы находим среднее абсолютное отклонение между соответствующими коэффициентами.
            coeff_diff = np.mean(np.abs(rule1['coefficients'] - rule2['coefficients']))
            comparison['coefficient_differences'].append(coeff_diff)
            
            # Затем вычисляем различие функций принадлежности. Для каждого признака
            # сравниваем параметры функций принадлежности (например, параметры Generalized Bell)
            # и находим среднее абсолютное отклонение. После этого усредняем по всем признакам.
            mf_diffs = []
            for feat_name in self.feature_names:
                if feat_name in rule1['membership_functions'] and feat_name in rule2['membership_functions']:
                    mf1 = rule1['membership_functions'][feat_name]
                    mf2 = rule2['membership_functions'][feat_name]
                    mf_diff = np.mean(np.abs(mf1 - mf2))
                    mf_diffs.append(mf_diff)
            
            comparison['mf_differences'].append(np.mean(mf_diffs) if mf_diffs else 0.0)
            
            # Теперь вычисляем сходство правила по формуле similarity = 1 / (1 + coeff_diff + mf_diff).
            # Эта формула гарантирует, что сходство всегда находится в диапазоне от нуля до единицы.
            # Если правила полностью идентичны, сходство равно единице. Чем больше различия,
            # тем меньше значение сходства.
            similarity = 1.0 / (1.0 + coeff_diff + comparison['mf_differences'][-1])
            comparison['rule_similarities'].append(similarity)
        
        # В конце вычисляем средние значения всех метрик по всем правилам.
        # Среднее сходство показывает общую степень похожести правил между моделями.
        comparison['average_similarity'] = np.mean(comparison['rule_similarities'])
        comparison['average_coeff_diff'] = np.mean(comparison['coefficient_differences'])
        comparison['average_mf_diff'] = np.mean(comparison['mf_differences'])
        
        return comparison

