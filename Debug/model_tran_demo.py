import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as pyplot
from bayes_opt import BayesianOptimization


class XGB_Model:
    def __init__(self, prifix,is_cv:bool=True,train_file_name:str=None):
        self.prifix = prifix
        if train_file_name is None:
            self.train_file_name = prifix + 'train.libsvm'
        else:
            self.train_file_name = train_file_name
        self.val_file_name = prifix + 'val.libsvm'
        self.test_file_name = prifix + 'test.libsvm'
        self.dtrain = xgb.DMatrix(self.train_file_name + "?format=libsvm")  # load sample
        self.dval = xgb.DMatrix(self.val_file_name + "?format=libsvm")  # load sample
        self.dtest = xgb.DMatrix(self.test_file_name + "?format=libsvm")  # load sample
        self.is_cv = is_cv

    def xgb_evaluate(seft, max_depth, eta, gamma, colsample_bytree, learning_rate, min_child_weight, scale_pos_weight,
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
        if (seft.is_cv):
            cv_result = xgb.cv(
                params,
                seft.dtrain,
                num_boost_round=100,
                nfold=10,
                stratified=True,
                early_stopping_rounds=50,
                seed=42,
                verbose_eval=False
            )
            return cv_result["test-auc-mean"].max()
        else:
            model = xgb.train(
                params,
                seft.dtrain,
                num_boost_round=300,
                evals=[(seft.dtrain, "train"), (seft.dval, "eval")],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            preds = model.predict(seft.dval)
            auc = roc_auc_score(seft.dval.get_label(), preds)
            return auc

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
            verbose=2
        )

        # # 开始优化
        optimizer.maximize(
            init_points=10,  # 随机探索点
            n_iter=250  # 贝叶斯优化迭代次数
        )

        # 输出最优结果
        # print("Best result:", optimizer.max)

        # 获取最优超参数
        best_params = optimizer.max['params']
        # best_params = {'colsample_bytree': 1.0, 'eta': 0.1, 'gamma': 0.0, 'learning_rate': 0.3, 'max_depth': 3.0,
        #                'min_child_weight': 4.615155109848909, 'reg_alpha': 2.134236173880166, 'reg_lambda': 5.0,
        #                'scale_pos_weight': 5.0, 'subsample': 1.0}
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_child_weight'] = int(best_params['min_child_weight'])

        # 👇 获取最佳轮数（用 xgb.cv 再跑一次，为了拿到 best_iteration）
        cv_result = xgb.cv(
            best_params,
            self.dtrain,
            num_boost_round=200,
            nfold=5,
            stratified=True,
            early_stopping_rounds=50,
            seed=42,
            verbose_eval=False
        )
        best_num_round = cv_result.shape[0]  # 最优轮数 = best_iteration
        print("Best params: ", optimizer.max)
        print("Best num_round: ", best_num_round)
        return best_params, best_num_round

    def model_tuning(self,params,num_round):
        evals_result = {}
        params = params
        params['device'] = 'cuda'
        params['tree_method'] = 'hist'
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'auc'
        params['max_depth'] = int(params['max_depth'])
        bst = xgb.train(
            params,
            self.dtrain,
            num_boost_round=num_round,
            evals=[(self.dtrain, "train"), (self.dval, "val")],
            early_stopping_rounds=10,
            evals_result=evals_result,
            verbose_eval=True
        )
        # 绘制学习曲线
        pyplot.figure(figsize=(10, 6))
        pyplot.plot(evals_result['train']['auc'], label='Train AUC')
        pyplot.plot(evals_result['val']['auc'], label='Validation AUC')
        pyplot.xlabel('Boosting Rounds')
        pyplot.ylabel('AUC')
        pyplot.title('Training Progress with Early Stopping')
        pyplot.legend()
        pyplot.grid(True)
        pyplot.savefig(self.prifix + "model.jpg")

        bst.save_model(self.prifix + "model.json")
        # load model
        model = xgb.Booster()
        model.load_model(self.prifix + "model.json")

        # 测试集评估
        y_pred = model.predict(self.dtest, iteration_range=(0, model.best_iteration + 1))

        if hasattr(model, 'best_iteration'):
            print(f"Best iteration: {model.best_iteration}")
        else:
            print("未启用早停法或版本不兼容")

        if hasattr(model, 'best_score'):
            print(f"Best Validation AUC: {model.best_score:.4f}")
        else:
            print("未启用早停法或版本不兼容")
        y_true = self.dtest.get_label()
        print(f"Test AUC: {roc_auc_score(y_true, y_pred):.4f}")
        y_pred_labels = [1 if x > 0.67 else 0 for x in y_pred]
        print("Accuracy:", accuracy_score(y_true, y_pred_labels))
        print("\nClassification Report:\n", classification_report(y_true, y_pred_labels))
if __name__ == "__main__":

    xgb_model = XGB_Model("T1P_sell_",False,'T1P_sell_smote_train.libsvm')
    best_params, best_num_round = xgb_model.bayyesian_optimize()
    # best_params = {'colsample_bytree': 1.0, 'eta': 1.0, 'gamma': 5.0, 'learning_rate': 0.01, 'max_depth': 4.281075764518275, 'min_child_weight': 2.2218856933808704, 'reg_alpha': 1.329714275909306, 'reg_lambda': 4.5304365393107355, 'scale_pos_weight': 1.0, 'subsample': 0.5}
    xgb_model.model_tuning(best_params,best_num_round)