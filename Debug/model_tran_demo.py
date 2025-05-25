import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as pyplot


train_file_name = 'T1_buy_train.libsvm'
val_file_name = 'T1_buy_val.libsvm'
test_file_name = 'T1_buy_test.libsvm'
prifix = 'T1_buy_'
# X_train, y_train = load_svmlight_file(train_file_name)
# X_val, y_val = load_svmlight_file(val_file_name)
# X_test, y_test = load_svmlight_file(test_file_name)
# X_train = X_train.toarray()
# smote = SMOTE(random_state=42)  # 不要100%平衡
# enn = EditedNearestNeighbours(n_neighbors=3)
# smote_enn = SMOTEENN(smote=smote, enn=enn)
# smote_tomek = SMOTETomek(random_state=42)
#
# X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
# dump_svmlight_file(X_resampled, y_resampled, prifi x +"somteenn_train.libsvm")

# load sample file & train model
# dtrain = xgb.DMatrix(prifix+"somteenn_train.libsvm" + "?format=libsvm")  # load sample
dtrain = xgb.DMatrix(train_file_name + "?format=libsvm")  # load sample
dval = xgb.DMatrix(val_file_name + "?format=libsvm")  # load sample
dtest = xgb.DMatrix(test_file_name + "?format=libsvm")  # load sample
# param = {'device': 'cuda', 'tree_method': 'hist', 'eta': 0.3, 'objective': 'binary:logistic','max_depth': 2,
#          'eval_metric': 'auc', 'reg_alpha':0.1,'reg_lambda':1.0}
# param = {'colsample_bytree': 0.4329211314861462, 'eta': 0.8442373851277825, 'gamma': 1.4581511825290598, 'learning_rate': 0.2453380070530987, 'max_depth': 3, 'min_child_weight': 5.916319324973441, 'reg_alpha': 0.3244133249128073, 'reg_lambda': 0.7653459681561497, 'scale_pos_weight': 2.868823840171599, 'subsample': 0.662752746276055}
param = {'colsample_bytree': 1.0, 'eta': 0.1, 'gamma': 0.0, 'learning_rate': 0.3, 'max_depth': 3.0, 'min_child_weight': 4.615155109848909, 'reg_alpha': 2.134236173880166, 'reg_lambda': 5.0, 'scale_pos_weight': 5.0, 'subsample': 1.0}
param['device'] = 'cuda'
param['tree_method'] = 'hist'
param['objective'] = 'binary:logistic'
param['eval_metric'] = 'auc'
param['max_depth'] = int(param['max_depth'])

evals_result = {}

bst = xgb.train(
    param,
    dtrain,
    num_boost_round=78,
    evals=[(dtrain, "train"), (dval, "val")],
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
pyplot.savefig(prifix + "model.jpg")

bst.save_model(prifix + "model.json")

# load model
model = xgb.Booster()
model.load_model(prifix + "model.json")

# 测试集评估
y_pred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

if hasattr(model, 'best_iteration'):
    print(f"Best iteration: {model.best_iteration}")
else:
    print("未启用早停法或版本不兼容")

if hasattr(model, 'best_score'):
    print(f"Best Validation AUC: {model.best_score:.4f}")
else:
    print("未启用早停法或版本不兼容")
y_true = dtest.get_label()
print(f"Test AUC: {roc_auc_score(y_true, y_pred):.4f}")
y_pred_labels = [1 if x > 0.67 else 0 for x in y_pred]
print("Accuracy:", accuracy_score(y_true, y_pred_labels))
print("\nClassification Report:\n", classification_report(y_true, y_pred_labels))