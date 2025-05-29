import xgboost as xgb
import json
import shap
import pingouin
import pandas as pd

import matplotlib.pyplot as pyplot
from sklearn.datasets import load_svmlight_file

from tqdm import tqdm

def partial_correlation(shap_df, y_target):
    parshap = {}
    y_target_series = pd.Series(y_target, name='target')
    for feature in tqdm(shap_df.columns):
        # 合并目标变量与SHAP值，排除当前特征的其他特征作为协变量
        df = pd.concat([shap_df.drop(feature, axis=1), shap_df[feature], y_target_series], axis=1)
        res = pingouin.partial_corr(data=df, x=feature,  y='target',
                                   covar=list(shap_df.columns.drop(feature)))
        parshap[feature] = res['r'].values[0]
    return pd.Series(parshap)
if __name__ == "__main__":
    feature_meta = {}

    meta = json.load(open("T2_buy_feature.meta", "r"))
    model = xgb.XGBClassifier()
    model.load_model("T2_buy_model.json")
    # predict
    importances = model.feature_importances_
    # print(importances)
    features_index = []
    for i in range(importances.shape[0]):
        if importances[i] > 0:
            features_index.append(i)
            print(i,importances[i])
    cols = [k for k, v in meta.items()]
    keys = [k for k, v in meta.items() if v not in features_index]

    print("keys", keys)
    importances_keys = [k for k, v in meta.items() if v in features_index]
    print("importances_keys", importances_keys)

    X_train, y_train = load_svmlight_file("T2_buy_train.libsvm")
    X_val, y_val = load_svmlight_file("T2_buy_val.libsvm")
    X_test, y_test = load_svmlight_file("T2_buy_test.libsvm")

    explainer = shap.TreeExplainer(model)
    shap_train = explainer.shap_values(X_train)
    shap_val = explainer.shap_values(X_val)
    shap_test = explainer.shap_values(X_test)

    shap_train_df = pd.DataFrame(shap_train, columns=cols)
    # # shap_train_col_df = pd.DataFrame(shap_train, columns=cols)
    shap_val_df = pd.DataFrame(shap_val, columns=cols)
    shap_test_df = pd.DataFrame(shap_test, columns=cols)
    # # shap_test_col_df = pd.DataFrame(shap_test, columns=cols)

    parshap_train = partial_correlation(shap_train_df, y_train)
    parshap_val = partial_correlation(shap_val_df, y_val)
    parshap_test = partial_correlation(shap_test_df, y_test)
    print('\n################# Print tran parshap')
    print(parshap_train.dropna().sort_values())
    print('\n################# Print val parshap')
    print(parshap_val.dropna().sort_values())
    print('\n################# Print test parshap')
    print(parshap_test.dropna().sort_values())
    parshap_diff = pd.Series(parshap_val - parshap_train, name='parshap_diff')
    print('\n################# Print val parshap_diff')  # 打印parshap差异
    # parshap_diff.sort_values() [k for k, v in meta.items() if v not in features_index]
    print(parshap_diff.dropna().sort_values())

    print('\n################# Print test parshap_diff')  # 打印parshap差异
    parshap_diff = pd.Series(parshap_test - parshap_train, name='parshap_diff')
    # parshap_diff.sort_values() [k for k, v in meta.items() if v not in features_index]
    print(parshap_diff.dropna().sort_values())

    # t1_buy_loaded_model = xgb.XGBClassifier()
    # t1_buy_loaded_model.load_model('T1buymodel.json')
    # plot_importance(model)
    # pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    # pyplot.savefig('T1buymodel_feature_importances.jpg')
