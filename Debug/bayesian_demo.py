import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 加载示例数据
dtrain = xgb.DMatrix("T1_buy_train.libsvm?format=libsvm")  # load sample
dval = xgb.DMatrix("T1_buy_val.libsvm?format=libsvm")  # load sample
dtest = xgb.DMatrix("T1_buy_test.libsvm?format=libsvm")  # load sample

# 定义目标函数
def xgb_evaluate(max_depth,eta, gamma, colsample_bytree, learning_rate, min_child_weight,scale_pos_weight,subsample,reg_lambda,reg_alpha):
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
    # model = xgb.train(
    #     params,
    #     dtrain,
    #     num_boost_round=100,
    #     evals=[(dtrain, "train"), (dval, "eval")],
    #     early_stopping_rounds=50,
    #     verbose_eval=False
    # )
    # preds = model.predict(dval)
    # auc = roc_auc_score(dval.get_label(), preds)
    # return auc
    cv_result = xgb.cv(
        params,
        dtrain,
        num_boost_round=100,
        nfold=10,
        stratified=True,
        early_stopping_rounds=50,
        seed=42,
        verbose_eval=False
    )
    return cv_result["test-auc-mean"].max()


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
    f=xgb_evaluate,
    pbounds=param_bounds,
    random_state=42,
    verbose=2
)

# 开始优化
optimizer.maximize(
    init_points=10,  # 随机探索点
    n_iter=250        # 贝叶斯优化迭代次数
)

# 输出最优结果
# print("Best result:", optimizer.max)

# 获取最优超参数
best_params = optimizer.max['params']
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_child_weight'] = int(best_params['min_child_weight'])

# 👇 获取最佳轮数（用 xgb.cv 再跑一次，为了拿到 best_iteration）
cv_result = xgb.cv(
    best_params,
    dtrain,
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
