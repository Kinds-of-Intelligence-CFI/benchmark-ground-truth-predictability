import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# --- GENERIC FUNCTIONS ---


# define generic function to evaluate a predictive method (such as LogisticRegression)
def evaluate_predictive_method(df_train, df_test, features_cols, response_col, predictive_method=LogisticRegression,
                               return_trained_method=False, return_instance_level_predictions=False,
                               compute_accuracy=False,
                               trained_method=None, binary=True, return_brier_average=True, **kwargs):
    if len(features_cols) == 1:
        if type(df_train[features_cols[0]].iloc[0]) == list:
            # todo check if this is ever used
            df_train[features_cols[0]] = df_train[features_cols[0]].apply(lambda x: np.array(x))
            df_test[features_cols[0]] = df_test[features_cols[0]].apply(lambda x: np.array(x))
        if type(df_train[features_cols[0]].iloc[0]) == np.ndarray:
            # if the features are already a numpy array, then don't convert to numpy array
            features_train = np.array(list(df_train[features_cols[0]].values))
            features_test = np.array(list(df_test[features_cols[0]].values))
        else:
            # features are not a list or a numpy array, so assume it is a single number
            # traditional one
            features_train = df_train[features_cols].to_numpy()
            features_test = df_test[features_cols].to_numpy()
    else:
        # traditional one
        features_train = df_train[features_cols].to_numpy()
        features_test = df_test[features_cols].to_numpy()

    if response_col in features_cols:
        raise ValueError("response_col must not be in features_cols")

    labels_train = df_train[response_col]
    labels_test = df_test[response_col]

    return _evaluate_predictive_method_from_arrays(features_train, labels_train, features_test, labels_test,
                                                   predictive_method, return_trained_method,
                                                   return_instance_level_predictions,
                                                   compute_accuracy,
                                                   trained_method, binary=binary,
                                                   return_brier_average=return_brier_average, **kwargs)


def _evaluate_predictive_method_from_arrays(features_train, labels_train, features_test, labels_test,
                                            predictive_method=LogisticRegression, return_trained_method=False,
                                            return_instance_level_predictions=False, compute_accuracy=False,
                                            trained_method=None,
                                            binary=True, return_brier_average=True, **kwargs):
    if trained_method is not None:
        method_instance = trained_method
    else:
        # fit logistic regression using training features and the agent col as response
        method_instance = predictive_method(**kwargs)
        method_instance.fit(features_train, labels_train)

    if binary:
        # evaluate on the test set
        y_pred = method_instance.predict_proba(features_test)[:, 1]

        BrierScore, Calibration, Refinement = brierDecomp(y_pred, labels_test,
                                                          return_brier_average=return_brier_average)
        # compute the ROC AUC using sklearn
        if not (sum(labels_test) == 0 or sum(labels_test) == len(labels_test)):
            roc_auc = roc_auc_score(labels_test, y_pred)
        else:
            roc_auc = np.nan

        if compute_accuracy:
            # compute accuracy by thresholding at 0.5
            y_pred_binary = y_pred > 0.5
            instance_level_success = y_pred_binary == labels_test
            accuracy = np.mean(instance_level_success)

        # make a dict
        return_dict = {"BrierScore": BrierScore, "Calibration": Calibration, "Refinement": Refinement,
                       "roc_auc": roc_auc}
    else:
        y_pred = method_instance.predict(features_test)
        instance_level_success = y_pred == labels_test
        accuracy = np.mean(instance_level_success)
        return_dict = {}

    if return_instance_level_predictions:
        return_dict["instance_level_predictions"] = y_pred
    if compute_accuracy:
        return_dict["accuracy"] = accuracy
    if return_trained_method:
        return_dict["method_instance"] = method_instance
    return return_dict


def brierDecomp(preds, outs, return_brier_average=True):
    brier = (preds - outs) ** 2
    if return_brier_average:
        brier = 1 / len(preds) * sum(brier)
    ## bin predictions
    bins = np.linspace(0, 1, 11)
    binCenters = (bins[:-1] + bins[1:]) / 2
    binPredInds = np.digitize(preds, binCenters)
    binnedPreds = bins[binPredInds]

    binTrueFreqs = np.zeros(10)
    binPredFreqs = np.zeros(10)
    binCounts = np.zeros(10)

    for i in range(10):
        idx = (preds >= bins[i]) & (preds < bins[i + 1])

        binTrueFreqs[i] = np.sum(outs[idx]) / np.sum(idx) if np.sum(idx) > 0 else 0
        # print(np.sum(outs[idx]), np.sum(idx), binTrueFreqs[i])
        binPredFreqs[i] = np.mean(preds[idx]) if np.sum(idx) > 0 else 0
        binCounts[i] = np.sum(idx)

    calibration = np.sum(binCounts * (binTrueFreqs - binPredFreqs) ** 2) / np.sum(binCounts) if np.sum(
        binCounts) > 0 else 0
    refinement = np.sum(binCounts * (binTrueFreqs * (1 - binTrueFreqs))) / np.sum(binCounts) if np.sum(
        binCounts) > 0 else 0
    # Compute refinement component
    # refinement = brier - calibration
    return brier, calibration, refinement
