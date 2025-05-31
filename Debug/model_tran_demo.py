import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, classification_report, roc_curve, \
    auc
import matplotlib.pyplot as pyplot
from bayes_opt import BayesianOptimization
import numpy as np
import logging
import json
import shap
import pingouin
import pandas as pd
from tqdm import tqdm

# 创建logger对象
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 设置日志级别

# 创建文件处理器
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)  # 文件日志级别

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # 控制台日志级别

# 创建日志格式
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加处理器到logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

dir_path = 'E:\\data\\BTC\\'
class XGB_Model:
    def __init__(self, prifix, is_cv: bool = True, train_file_name: str = None):
        self.path = dir_path + prifix + "\\"
        self.prifix =  self.path + prifix + "_"
        if train_file_name is None:
            self.train_file_name = self.prifix + 'train.libsvm'
        else:
            self.train_file_name = train_file_name
        self.val_file_name = self.prifix + 'val.libsvm'
        self.test_file_name = self.prifix + 'test.libsvm'
        self.dtrain = xgb.DMatrix(self.train_file_name + "?format=libsvm")  # load sample
        self.dval = xgb.DMatrix(self.val_file_name + "?format=libsvm")  # load sample
        self.dtest = xgb.DMatrix(self.test_file_name + "?format=libsvm")  # load sample
        self.is_cv = is_cv
        self.best_model = None  # 全局变量保存最优模型
        self.best_score = -np.inf
        self.evals_result = {}
        self.best_iteration = 300
        self.best_params = {}

    def xgb_evaluate(self, max_depth, eta, gamma, colsample_bytree, learning_rate, min_child_weight, scale_pos_weight,
                     subsample, reg_lambda, reg_alpha):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": int(max_depth),
            "eta": eta,
            "gamma": gamma,
            "colsample_bytree": colsample_bytree,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "scale_pos_weight": scale_pos_weight,
            "subsample": subsample,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "verbosity": 0
        }
        if self.is_cv:
            cv_result = xgb.cv(
                params,
                self.dtrain,
                num_boost_round=100,
                nfold=10,
                stratified=True,
                early_stopping_rounds=50,
                seed=42,
                verbose_eval=False
            )
            score = cv_result["test-auc-mean"].max()
            if score > self.best_score:  # 更新最优模型
                self.best_params = params
                self.best_score = score
            return score
        else:
            evals_result = {}
            model = xgb.train(
                params,
                self.dtrain,
                num_boost_round=1000,
                evals=[(self.dtrain, "train"), (self.dval, "eval")],
                evals_result=evals_result,
                early_stopping_rounds=50,
                verbose_eval=False
            )
            if model.best_score > self.best_score:  # 更新最优模型
                self.best_model = model
                self.best_score = model.best_score
                self.evals_result = evals_result
                self.best_iteration = model.best_iteration
            return model.best_score

    def bayyesian_optimize(self):
        # 设置超参数空间
        param_bounds = {
            "max_depth": (1, 10),
            "eta": (0.1, 1),
            "gamma": (0, 5),
            "colsample_bytree": (0.3, 1),
            "learning_rate": (0.01, 0.3),
            "min_child_weight": (1, 10),
            "scale_pos_weight": (1, 5),
            "subsample": (0.5, 1),
            "reg_lambda": (0, 5),
            "reg_alpha": (0, 5),
        }
        # 贝叶斯优化器
        optimizer = BayesianOptimization(
            f=self.xgb_evaluate,
            pbounds=param_bounds,
            random_state=42,
            verbose=0
        )

        # # 开始优化
        optimizer.maximize(
            init_points=10,  # 随机探索点
            n_iter=250  # 贝叶斯优化迭代次数
        )

        if self.is_cv:
            evals_result = {}
            model = xgb.train(
                self.best_params,
                self.dtrain,
                num_boost_round=100,
                evals=[(self.dtrain, "train"), (self.dval, "eval")],
                evals_result=evals_result,
                early_stopping_rounds=50,
                verbose_eval=False
            )
            self.best_model = model
            self.evals_result = evals_result
            self.best_iteration = model.best_iteration

        # 输出最优结果
        # print("Best result:", optimizer.max)

    def model_tuning(self):
        evals_result = self.evals_result

        # 绘制学习曲线
        pyplot.figure(figsize=(10, 6))
        pyplot.plot(evals_result['train']['auc'], label='Train AUC')
        pyplot.plot(evals_result['eval']['auc'], label='Validation AUC')
        pyplot.xlabel('Boosting Rounds')
        pyplot.ylabel('AUC')
        pyplot.title('Training Progress with Early Stopping')
        pyplot.legend()
        pyplot.grid(True)
        pyplot.savefig(self.prifix + "model.jpg")

        bst = self.best_model

        bst.save_model(self.prifix + "model.json")
        # load model
        model = xgb.Booster()
        model.load_model(self.prifix + "model.json")

        print(self.best_iteration, model.best_iteration, model.best_score)
        eval_auc, index = max((value, index) for index, value in enumerate(evals_result['eval']['auc']))
        train_auc = evals_result['train']['auc'][index]
        # 测试集评估
        y_pred = model.predict(self.dtest, iteration_range=(0, model.best_iteration + 1))

        if hasattr(model, 'best_iteration'):
            logger.info("Best iteration:  %d", model.best_iteration)
        else:
            print("未启用早停法或版本不兼容")

        if hasattr(model, 'best_score'):
            logger.info("Best Validation AUC:  %f", model.best_score)
        else:
            print("未启用早停法或版本不兼容")
        y_true = self.dtest.get_label()
        logger.info("Train AUC:  %f", train_auc)
        test_auc = roc_auc_score(y_true, y_pred)
        logger.info("Test AUC:  %f",model.best_score)
        y_pred_labels = [1 if x > 0.67 else 0 for x in y_pred]
        logger.info("Accuracy: %f", accuracy_score(y_true, y_pred_labels))
        # print("\nClassification Report:\n", classification_report(y_true, y_pred_labels))

        if train_auc - test_auc > 0.1 or train_auc - eval_auc > 0.1:
            meta = json.load(open(self.prifix + "feature.meta", "r"))
            class_model = xgb.XGBClassifier()
            class_model.load_model(self.prifix + "model.json")
            features_index = []
            importances = class_model.feature_importances_
            for i in range(importances.shape[0]):
                if importances[i] > 0:
                    features_index.append(i)
                    # print(i, importances[i])
            cols = [k for k, v in meta.items()]
            keys = [k for k, v in meta.items() if v not in features_index]
            logger.info("keys:\n %s", keys)
            importances_keys = [k for k, v in meta.items() if v in features_index]
            logger.info("importances_keys:\n %s", importances_keys)
            X_train, y_train = load_svmlight_file(self.train_file_name)
            X_val, y_val = load_svmlight_file(self.val_file_name)
            X_test, y_test = load_svmlight_file(self.test_file_name)
            explainer = shap.TreeExplainer(class_model)
            shap_train = explainer.shap_values(X_train)
            shap_val = explainer.shap_values(X_val)
            shap_test = explainer.shap_values(X_test)

            shap_train_df = pd.DataFrame(shap_train, columns=cols)
            shap_val_df = pd.DataFrame(shap_val, columns=cols)
            shap_test_df = pd.DataFrame(shap_test, columns=cols)

            parshap_train = self.partial_correlation(shap_train_df, y_train)
            parshap_val = self.partial_correlation(shap_val_df, y_val)
            parshap_test = self.partial_correlation(shap_test_df, y_test)
            logger.info("Print train parshap:\n %s", parshap_train.dropna().sort_values())
            logger.info("Print val parshap:\n %s", parshap_val.dropna().sort_values())
            logger.info("Print test parshap:\n %s",parshap_test.dropna().sort_values())
            parshap_diff = pd.Series(parshap_val - parshap_train, name='parshap_diff')
            logger.info("Print val parshap_diff:\n %s", parshap_diff.dropna().sort_values())
            parshap_diff = pd.Series(parshap_test - parshap_train, name='parshap_diff')
            logger.info("Print test parshap_diff:\n %s", parshap_diff.dropna().sort_values())

    def partial_correlation(self, shap_df, y_target):
        parshap = {}
        y_target_series = pd.Series(y_target, name='target')
        for feature in tqdm(shap_df.columns):
            # 合并目标变量与SHAP值，排除当前特征的其他特征作为协变量
            df = pd.concat([shap_df.drop(feature, axis=1), shap_df[feature], y_target_series], axis=1)
            res = pingouin.partial_corr(data=df, x=feature, y='target',
                                        covar=list(shap_df.columns.drop(feature)))
            parshap[feature] = res['r'].values[0]
        return pd.Series(parshap)


def exec(prifix, is_cv, train_file_name):
    logger.info("prifix: %s, is_cv: %s, train_file_name: %s, flag: %s", prifix, is_cv, train_file_name, 'start')
    xgb_model = XGB_Model(prifix, is_cv, train_file_name)
    xgb_model.bayyesian_optimize()
    xgb_model.model_tuning()
    logger.info("prifix: %s, is_cv: %s, train_file_name: %s, flag: %s", prifix, is_cv, train_file_name, 'end')


def batch_exec(prifix):
    exec(prifix, False, prifix + 'smote_tomek_train.libsvm')
    exec(prifix, False, prifix+'smote_train.libsvm')
    exec(prifix, False, prifix+'somteenn_train.libsvm')
    exec(prifix, False, prifix+'train.libsvm')
    exec(prifix, True, prifix+'smote_tomek_train.libsvm')
    exec(prifix, True, prifix+'smote_train.libsvm')
    exec(prifix, True, prifix+'somteenn_train.libsvm')
    exec(prifix, True, prifix+'train.libsvm')


if __name__ == "__main__":
    # 0.7363 0.6471
    batch_exec("T1_buy_selected")
    batch_exec("T1_buy")
    batch_exec("T1_sell_selected")
    batch_exec("T1_sell")
    batch_exec("T1P_buy_selected")
    batch_exec("T1P_buy")
    batch_exec("T1P_sell_selected")
    batch_exec("T1P_sell")
    batch_exec("T2_buy_selected")
    batch_exec("T2_buy")
    batch_exec("T2_sell_selected")
    batch_exec("T2_sell")
    batch_exec("T2S_buy_selected")
    batch_exec("T2S_buy")
    batch_exec("T2S_sell_selected")
    batch_exec("T2S_sell")
