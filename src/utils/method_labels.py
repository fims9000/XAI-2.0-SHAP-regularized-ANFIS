"""
Общие константы для наименования метода
Baseline-Masked Feature Sensitivity Regularization.
"""

METHOD_NAME_EN = "Baseline-Masked Feature Sensitivity Regularization"
METHOD_NAME_RU = (
    "Регуляризация чувствительности признаков "
    "с маскированием по базовым значениям"
)
METHOD_ALIAS_RU = "оценка чувствительности признаков"

METHOD_LABEL_COMPACT = f"{METHOD_NAME_EN}\n({METHOD_ALIAS_RU})"
METHOD_LABEL_INLINE = f"{METHOD_NAME_EN} ({METHOD_ALIAS_RU})"
METHOD_LABEL_LONG = f"{METHOD_NAME_EN}\n({METHOD_NAME_RU})"

METHOD_MODEL_LABEL = f"ANFIS + {METHOD_NAME_EN}\n({METHOD_ALIAS_RU})"
METHOD_MODEL_LABEL_SINGLE = f"ANFIS + {METHOD_NAME_EN}"
METHOD_MODEL_LABEL_RU = f"ANFIS с {METHOD_LABEL_INLINE}"

METHOD_ACRONYM = "BMFSR"
METHOD_LABEL_SHORT = METHOD_ACRONYM
METHOD_LABEL_SHORT_INLINE = f"{METHOD_ACRONYM} ({METHOD_ALIAS_RU})"
METHOD_MODEL_LABEL_SHORT = f"ANFIS+{METHOD_ACRONYM}"


