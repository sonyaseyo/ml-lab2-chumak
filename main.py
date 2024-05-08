import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, precision_recall_curve, roc_curve
import numpy as np
from matplotlib.cm import get_cmap

# -------------------- 1
file_path = 'KM-12-3.csv'
data = pd.read_csv(file_path)
print(data.head())
actual_values = data['GT']
model1_probs = data['Model_1_0']
model2_probs = data['Model_2_1']

# -------------------- 2
class_counts = data['GT'].value_counts()
print("\nКількість об’єктів кожного класу:")
print(class_counts)

# -------------------- 3
# a. Обчислення метрик для кожної моделі при різних значеннях порогу
def compute_metrics(data, threshold_step=0.2):
    metrics = {}

    for model in data.columns[1:]:
        metrics[model] = {}
        for threshold in np.arange(0, 1, threshold_step):
            predicted_class = (data[model] >= threshold).astype(int)
            accuracy = accuracy_score(data['GT'], predicted_class)
            precision = precision_score(data['GT'], predicted_class)
            recall = recall_score(data['GT'], predicted_class)
            f1 = f1_score(data['GT'], predicted_class)
            mcc = matthews_corrcoef(data['GT'], predicted_class)
            balanced_accuracy = balanced_accuracy_score(data['GT'], predicted_class)

            fpr, tpr, _ = roc_curve(data['GT'], data[model])
            auc_roc = roc_auc_score(data['GT'], data[model])

            precision_, recall_, _ = precision_recall_curve(data['GT'], data[model])
            auc_pr = np.trapz(recall_, precision_)

            metrics[model][threshold] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Matthews Correlation Coefficient': mcc,
                'Balanced Accuracy': balanced_accuracy,
                'Area Under Curve (ROC)': auc_roc,
                'Area Under Curve (PR)': auc_pr
            }

    return metrics


metrics = compute_metrics(data)

# b. Збудувати  на  одному  графіку  в  одній  координатній  системі графіки усіх обчислених метрик, відмітивши певним чином максимальне значення кожної з них')

def plot_metrics(metrics):
    plt.figure(figsize=(12, 8))

    cmap = get_cmap('tab10')

    for i, (model, thresholds) in enumerate(metrics.items()):
        model_color = cmap(i)
        for metric, values in thresholds.items():
            x = list(values.keys())
            y = list(values.values())
            plt.plot(x, y, label=f"{model} - {metric}", color=model_color)

    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs Threshold')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

plot_metrics(metrics)

# 4. Зробити висновки щодо якості моделей, визначити кращу модель.

# Найвищі значення метрик має модель Model_2_1, тому її можна вважати якісною,
# Загалом модель Model_2_1 має вищі показники за Model_1_0, а отже вважається кращою,


# -------------------- 5
def calculate_K(bday):
    day, month = bday.split('-')
    day = int(day)
    K = (day % 9)
    return K

bday = '26-12'
K = calculate_K(bday)
print("K =", K)

class_1_count = data['GT'].sum()
percent_to_remove = 50 + 5 * K
objects_to_remove = int(class_1_count * percent_to_remove / 100)
class_1_indices = data[data['GT'] == 1].index
objects_to_remove_indices = np.random.choice(class_1_indices, size=objects_to_remove, replace=False)
new_data = data.drop(objects_to_remove_indices)

print("к-кість об'єктів класу 1 в початковому наборі даних:", class_1_count)
print("к-кість об'єктів класу 1 у новому наборі даних:", new_data['GT'].sum())


# -------------------- 6
removed_percentage = (objects_to_remove / class_1_count) * 100

# к-кість об'єктів класу 1 та 0 після видалення
new_class_1_count = new_data['GT'].sum()
new_class_0_count = len(new_data) - new_class_1_count

print("відсоток видалених об'єктів класу 1: {:.2f}%".format(removed_percentage))
print("к-кість об'єктів класу 1 після видалення:", new_class_1_count)
print("к-кість об'єктів класу 0 після видалення:", new_class_0_count)


# -------------------- 7
new_metrics = compute_metrics(new_data) #а
plot_metrics(new_metrics) #б


print(objects_to_remove * 100)
print(removed_percentage)