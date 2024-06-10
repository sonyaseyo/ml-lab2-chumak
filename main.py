import pandas as pd
import numpy as np
import math
from sklearn.metrics import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def fdiv(a, b):
    return 1 if not b else a / b

epsilon = 0.0005

def roundf(val):
    precision = 3
    return round(val, precision)

# Створити директорію для збереження графіків
output_dir = 'graphs'
os.makedirs(output_dir, exist_ok=True)

# -------------------- 1
file_path = 'KM-12-3.csv'
data = pd.read_csv(file_path)
print(data.head())

# -------------------- 2
print("\nКількість об’єктів кожного класу:")
print(data['GT'].value_counts())

data['Model_1_1'] = abs(data['Model_1_0'] - 1)
data['Model_2_0'] = abs(data['Model_2_1'] - 1)
data = data.reindex(columns=sorted(data.columns))
data

# -------------------- 3
threshold = np.linspace(-epsilon, 1 + epsilon, int(1 / epsilon))
fact_value = data.columns[0]
models = data.columns[2::2].values

def calc(df, model, threshold):
    model_exmp = dict()

    for t in tqdm(threshold):
        TP = len(df[(df[fact_value] == 1) & (df[model] >= t)])
        FP = len(df[(df[fact_value] == 0) & (df[model] >= t)])
        FN = len(df[(df[fact_value] == 1) & (df[model] < t)])
        TN = len(df[(df[fact_value] == 0) & (df[model] < t)])

        accuracy = fdiv((TP + TN), (TP + TN + FP + FN))
        precision = fdiv(TP, (TP + FP))
        recall = fdiv(TP, (TP + FN))
        f_score = fdiv(2 * precision * recall, (precision + recall))

        try:
            MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        except ZeroDivisionError:
            MCC = 0

        sensitivity = fdiv(TP, (TP + FN))
        specificity = fdiv(TN, (TN + FP))

        BA = (sensitivity + specificity) / 2

        J_statistics = sensitivity + specificity - 1

        model_exmp.update({'accuracy': model_exmp.get('accuracy', []) + [accuracy]})
        model_exmp.update({'precision': model_exmp.get('precision', []) + [precision]})
        model_exmp.update({'recall': model_exmp.get('recall', []) + [recall]})
        model_exmp.update({'f_score': model_exmp.get('f_score', []) + [f_score]})
        model_exmp.update({'MCC': model_exmp.get('MCC', []) + [MCC]})
        model_exmp.update({'specificity': model_exmp.get('specificity', []) + [specificity]})
        model_exmp.update({'sensitivity': model_exmp.get('sensitivity', []) + [sensitivity]})
        model_exmp.update({'BA': model_exmp.get('BA', []) + [BA]})
        model_exmp.update({'J_statistics': model_exmp.get('J_statistics', []) + [J_statistics]})

    FPR = [*map(lambda x: 1 - x, model_exmp['specificity'])]
    TPR = model_exmp['sensitivity']

    PRC_AUC = auc(model_exmp['recall'], model_exmp['precision'])
    ROC_AUC = auc(FPR, TPR)

    Joudens_index = max(model_exmp['J_statistics'])

    model_exmp["Youden's index"] = Joudens_index
    model_exmp['PRC_AUC'] = PRC_AUC
    model_exmp['ROC_AUC'] = ROC_AUC

    return model_exmp

model_1 = dict({'name': models[0], 'model': calc(data, models[0], threshold)})
model_2 = dict({'name': models[1], 'model': calc(data, models[1], threshold)})

def calc_metrics(model_1, model_2):
    for model in (model_1, model_2):
        print(f"Model: {model['name']}", end='\n\n')

        for k,v in model['model'].items():
            if k not in ['specificity', 'sensitivity']:
                print(f'{k} (mean): {roundf(sum(v) / len(v))}') if isinstance(v, list) else print(f'{k}: {roundf(v)}')

        print('\n\n') if model != model_2 else print('')

calc_metrics(model_1, model_2)

def viz_metrics(model_1, model_2):
    for model in (model_1, model_2):
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_iter = iter(color_cycle)
        for k,v in model['model'].items():
            if isinstance(v, list) and k not in ['specificity', 'sensitivity']:
                color = next(color_iter)
                plt.plot(threshold, v, label=str(k), linewidth=2.5, color=color)
                plt.plot(threshold[v.index(max(v))], max(v), marker='*', markersize=15, color=color)
                plt.xticks(np.arange(0, 1.01, .1))
                plt.yticks(np.arange(0, 1.01, .1))
        plt.title(model['name'])
        plt.legend()
        plt.savefig(f"{output_dir}/{model['name']}_metrics.png")
        plt.show()
        plt.close()

viz_metrics(model_1, model_2)

def plot_model(model, metric_threshold):
    name = model['name']
    mod = data[[fact_value, name]].sort_values(by=[name])
    model_values = sorted(list(zip(mod.value_counts().keys(), mod.value_counts().tolist())),
                          key=lambda x: float(x[0][1]))
    model_ones = dict([(el[0][1], el[1]) for el in model_values if el[0][0] == 1])
    model_zeros = dict([(el[0][1], el[1]) for el in model_values if el[0][0] == 0])

    counter = 0
    for k, v in model_ones.items():
        counter += model_ones.get(k)
        model_ones.update({k: counter})

    counter = 0
    for k, v in model_zeros.items():
        counter += model_zeros.get(k)
        model_zeros.update({k: counter})

    opt_metrics = dict()

    for metric, value in model['model'].items():
        if isinstance(value, list) and metric not in ['specificity', 'sensitivity']:
            pair = list(zip(threshold, value))
            pair = list(filter(lambda x: x[1] >= metric_threshold, pair))
            if pair:
                opt_metrics.update({metric: pair[0][0]})
            else:
                opt_metrics.update({metric: None})

    plt.figure(figsize=(15, 7.5))
    ax = plt.gca()

    def plot_opt_metrics(ax):
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_iter = iter(color_cycle)
        for metric, threshold in opt_metrics.items():
            color = next(color_iter)
            ax.plot(np.repeat(opt_metrics[metric], 10), np.linspace(0, list(model_ones.values())[-1], 10), color=color,
                    ls='--', label=metric)
            plt.legend()

    ax1 = plt.subplot(1, 2, 1)
    plt.plot(list(model_ones.keys()), list(model_ones.values()))
    plot_opt_metrics(ax1)
    plt.title(name)
    plt.xlabel('Значення оцінки класифікатора')
    plt.ylabel('Кількість true_label = 1')

    ax2 = plt.subplot(1, 2, 2)
    plt.plot(list(model_zeros.keys()), list(model_zeros.values()))
    plot_opt_metrics(ax2)
    plt.title(name)
    plt.xlabel('Значення оцінки класифікатора')
    plt.ylabel('Кількість true_label = 0')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}_model_plot.png")
    plt.show()
    plt.close()

plot_model(model_1, 0.95)
plot_model(model_2, 0.95)

def plot_prc(model_1, model_2):
    global opt
    for mod in (model_1, model_2):
        name, model = mod['name'], mod['model']
        precision, recall = model['precision'], model['recall']
        PRC = [*zip(precision, recall)]

        for pair in PRC:
            if roundf(abs(pair[0] - pair[1])) <= 0.03:
                opt = pair
                break

        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, label='PR-curve')
        avg = (opt[1] + opt[0]) / 2
        plt.plot(avg, avg, color='red', marker='o', markersize=15)
        plt.plot([0, 1], [0, 1], label='y = x', color='red')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {name}')
        plt.legend()
        plt.savefig(f"{output_dir}/{name}_prc_curve.png")
        plt.show()
        plt.close()
        print(f'{name} PRC-AUC: {roundf(auc(recall, precision))}')

plot_prc(model_1, model_2)

def plot_roc(model_1, model_2):
    for mod in (model_1, model_2):
        name, model = mod['name'], mod['model']
        FPR = [*map(lambda x: 1 - x, model['specificity'])]
        TPR = model['sensitivity']
        lst = list(map(lambda x: x[0]**2 + (1 - x[1])**2, list(zip(FPR, TPR))))
        opt = list(zip(FPR,TPR))[lst.index(min(lst))]
        plt.figure(figsize=(8,8))
        plt.xticks(np.arange(0, 1.01, .1))
        plt.yticks(np.arange(0, 1.01, .1))
        plt.xlabel('1 - Specificity (FPR)')
        plt.ylabel('Sensitivity (TPR)')
        plt.plot(FPR, TPR, label='ROC curve')
        plt.plot(opt[0], opt[1], color='red', marker='o', markersize=10)
        plt.plot(np.repeat(opt[0], 10), np.linspace(opt[0], 1, 10), color='red', ls='--', label='Youden Index')
        plt.plot([0,1], [0,1], ls='--', label='Random chance line')
        plt.title(f'ROC-curve for {name}')
        plt.legend()
        plt.savefig(f"{output_dir}/{name}_roc_curve.png")
        plt.show()
        plt.close()
        print(f'{name} ROC-AUC: {roundf(auc(FPR, TPR))}')

plot_roc(model_1, model_2)

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

print(objects_to_remove * 100)
print(removed_percentage)

model_1a = dict({'name': models[0], 'model': calc(new_data, models[0], threshold)})
model_2a = dict({'name': models[1], 'model': calc(new_data, models[1], threshold)})

calc_metrics(model_1a, model_2a)

viz_metrics(model_1a, model_2a)

plot_model(model_1a, 0.8)
plot_model(model_2a, 0.8)

plot_prc(model_1a, model_2a)

plot_roc(model_1a, model_2a)
