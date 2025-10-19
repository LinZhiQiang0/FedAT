import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    roc_auc_score,
    f1_score,
    recall_score,
)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt

# 读取数据集
# data = pd.read_excel(r"E:\bishe\features_1\sign_features_2.xlsx")
# data = pd.read_excel(r"C:\Users\2023\Desktop\EEG\s\7ge.csv")
# data = pd.read_csv(r"E:\bishe\features_1\channel_feature\F2.csv")
# 指定 Excel 文件路径
excel_file = "C:/Users/2023/Desktop/EEG/4eye.xlsx"

# 指定引擎来读取 Excel 文件
try:
    data = pd.read_excel(excel_file, engine="openpyxl")
    # 如果成功读取文件，可以继续进行后续操作
    print("Successfully read the Excel file.")
    # 进行其他操作，例如数据分析或处理
except ValueError as e:
    # 如果出现错误，则输出错误消息
    print("Error:", e)
# # 分离特征和标签
# X = data.iloc[2:-1, :-1]
# y = data.iloc[2:-1, -1]
# # 特征选择：第一行到最后一行，第一列到倒数第二列
# X = data.iloc[1:, :-1]
# # 标签选择：最后一列
# y = data.iloc[1:, -1]
# 特征选择：第二行到最后一行，第一列到倒数第二列
X = data.iloc[1:, :-1]

# 标签选择：第二行到最后一行的最后一列
y = data.iloc[1:, -1]

##将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
##创建一个 KNN 分类器
knn = KNeighborsClassifier()
knn_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
##定义一个参数网格，用于交叉验证选择最优参数
knn_param_grid = {
    "n_neighbors": np.arange(1, 10),
    "weights": ["uniform", "distance"],
    "algorithm": ["ball_tree", "kd_tree", "brute"],
    "p": [1, 2],
}
##使用交叉验证搜索最佳参数
knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=knn_cv)
knn_grid_search.fit(X_train, y_train)
##输出最佳参数和最佳得分
print("knn_Best parameters: ", knn_grid_search.best_params_)
print("knn_Best score: {:.2f}".format(knn_grid_search.best_score_))
##设置正向选择
knn_sfs = SFS(knn, k_features="best", forward=True, cv=knn_cv)
knn_sfs.fit(X, y)
##输出最优子集
print("Best feature subset: ", knn_sfs.k_feature_idx_)
##输出最优子集对应的特征名
selected_features = X.columns[list(knn_sfs.k_feature_idx_)]
print("Selected features: ", selected_features)
##使用选择的最优特征进行分类
X_train_fs = knn_sfs.transform(X_train)
X_test_fs = knn_sfs.transform(X_test)
knn_clf = KNeighborsClassifier(
    n_neighbors=knn_grid_search.best_params_["n_neighbors"],
    weights=knn_grid_search.best_params_["weights"],
    algorithm=knn_grid_search.best_params_["algorithm"],
    p=knn_grid_search.best_params_["p"],
)
knn_clf.fit(X_train_fs, y_train)
knn_y_pred = knn_clf.predict(X_test_fs)
knn_y_pred_proba = knn_clf.predict_proba(X_test_fs)[:, 1]
##计算分类器的准确率
knn_accuracy = accuracy_score(y_test, knn_y_pred)
print("knn_acc: {:.2f}%".format(knn_accuracy * 100))
##计算 ROC 曲线和 AUC 值
knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, knn_y_pred_proba, pos_label=1)
knn_auc = roc_auc_score(y_test, knn_y_pred_proba)
##计算 f1 和 recall
knn_f1 = f1_score(y_test, knn_y_pred)
knn_recall = recall_score(y_test, knn_y_pred)
print("knn_AUC:", knn_auc)
print("knn_F1:", knn_f1)
print("knn_Recall:", knn_recall)
#################################################RF
# 创建随机森林分类器
rf = RandomForestClassifier()
rf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# 设置参数范围
rf_param_grid = {
    "n_estimators": np.arange(500, 1000, 100),
    "max_depth": np.arange(5, 10, 1),
}
# 使用交叉验证搜索最佳参数
rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=rf_cv)
rf_grid_search.fit(X_train, y_train)
# 输出最佳参数和最佳得分
print("rf_Best parameters: ", rf_grid_search.best_params_)
print("rf_Best score: {:.2f}".format(rf_grid_search.best_score_))
# 设置正向选择
rf_sfs = SFS(rf, k_features="best", forward=True, cv=rf_cv)
rf_sfs.fit(X, y)
# 输出最优子集
print("Best feature subset: ", rf_sfs.k_feature_idx_)
# 输出最优子集对应的特征名
selected_features = X.columns[list(rf_sfs.k_feature_idx_)]
print("Selected features: ", selected_features)
# 使用选择的最优特征进行分类
X_train_fs = rf_sfs.transform(X_train)
X_test_fs = rf_sfs.transform(X_test)
rf_clf = RandomForestClassifier(
    n_estimators=rf_grid_search.best_params_["n_estimators"],
    max_depth=rf_grid_search.best_params_["max_depth"],
)
rf_clf.fit(X_train_fs, y_train)
y_pred = rf_clf.predict(X_test_fs)
rf_y_pred_proba = rf_clf.predict_proba(X_test_fs)[:, 1]
# 计算准确率
rf_acc = accuracy_score(y_test, y_pred)
print("rf_Accuracy: {:.2f}%".format(rf_acc * 100))
# 计算 ROC 曲线和 AUC 值
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_y_pred_proba, pos_label=1)
rf_auc = roc_auc_score(y_test, rf_y_pred_proba)
# 计算 f1 和 recall
rf_f1 = f1_score(y_test, y_pred)
rf_recall = recall_score(y_test, y_pred)
print("rf_AUC:", rf_auc)
print("rf_F1:", rf_f1)
print("rf_Recall:", rf_recall)
###########################################SVM
# 定义 SVM 模型
svm = SVC(kernel="rbf")
svm_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# 设置参数范围
svm_param_grid = {"C": np.arange(1, 10, 1), "gamma": np.arange(0.01, 0.1, 0.01)}
# 使用交叉验证搜索最佳参数
svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=svm_cv)
svm_grid_search.fit(X_train, y_train)
# 输出最佳参数和最佳得分
print("svm_Best parameters: ", svm_grid_search.best_params_)
print("svm_Best score: {:.2f}".format(svm_grid_search.best_score_))
# 设置正向选择
svm_sfs = SFS(svm, k_features="best", forward=True, cv=svm_cv)
svm_sfs.fit(X, y)
# 输出最优子集
print("Best feature subset: ", svm_sfs.k_feature_idx_)
# 输出最优子集对应的特征名
selected_features = X.columns[list(svm_sfs.k_feature_idx_)]
print("Selected features: ", selected_features)
# 使用选择的最优特征进行分类
X_train_fs = svm_sfs.transform(X_train)
X_test_fs = svm_sfs.transform(X_test)
svm_clf = SVC(
    kernel="rbf",
    C=svm_grid_search.best_params_["C"],
    gamma=svm_grid_search.best_params_["gamma"],
    probability=True,
)
svm_clf.fit(X_train_fs, y_train)
y_pred = svm_clf.predict(X_test_fs)
svm_y_pred_proba = svm_clf.predict_proba(X_test_fs)[:, 1]
# 计算准确率
svm_acc = accuracy_score(y_test, y_pred)
print("svm_Accuracy: {:.2f}%".format(svm_acc * 100))
# 计算 ROC 曲线和 AUC 值
svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, svm_y_pred_proba, pos_label=1)
svm_auc = roc_auc_score(y_test, svm_y_pred_proba)
# 计算 f1 和 recall
svm_f1 = f1_score(y_test, y_pred)
svm_recall = recall_score(y_test, y_pred)
print("svm_AUC:", svm_auc)
print("svm_F1:", svm_f1)
print("svm_Recall:", svm_recall)
# Plot ROC curves for both models
plt.figure()
plt.plot(svm_fpr, svm_tpr, color="red", lw=2, label="SVM (AUC = %0.2f)" % svm_auc)
plt.plot(knn_fpr, knn_tpr, color="blue", lw=2, label="KNN (AUC = %0.2f)" % knn_auc)
plt.plot(rf_fpr, rf_tpr, color="green", lw=2, label="RF (AUC = %0.2f)" % rf_auc)
plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()
