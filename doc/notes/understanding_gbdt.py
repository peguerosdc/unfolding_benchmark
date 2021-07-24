def predict_stages(estimators, X, scale):
    # https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/ensemble/_gradient_boosting.pyx#L190
    out = 0
    for tree in estimators:
        # tree.predict no hace conversiones especiales
        out += scale * tree.predict(X).reshape((X.shape[0], 1))
    return out

class GradientBoostingClassifier(Object):
    """
    Para clasificaciones binarias:
    self.loss_ = BinomialDeviance
        https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/ensemble/_gb_losses.py#L548
    """

    def predict_proba(self, X):
        raw_predictions = self.decision_function(X)
        return self.loss_._raw_prediction_to_proba(raw_predictions)

    def decision_function(self, X):
        # https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/ensemble/_gb.py#L1124
        return self._raw_predict(X)

    def _raw_predict(self, X):
        # https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/ensemble/_gb.py#L622
        """Return the sum of the trees raw predictions (+ init estimator)."""
        raw_predictions = self._raw_predict_init(X)
        predictions = raw_predictions + predict_stages(self.estimators_, X, self.learning_rate)
        return predictions

    def _raw_predict_init(self, X):
        # https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/ensemble/_gb.py#L607
        return self.loss_.get_init_raw_predictions(X, estimators)


class BinomialDeviance(object):

    def _raw_prediction_to_proba(self, raw_predictions):
        # https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/ensemble/_gb_losses.py#L635
        # Proba contiene dos entradas por elemento:
        # la entrada 1 es la probabilidad de clasificacion p_c
        # la entrada 0 es "1 - p_c"
        proba = []
        """
        np.expit(x) = función logística:
                        1/(1+exp(-x))
        """
        probability_of_class = np.expit(raw_predictions)
        proba[1] = probability_of_class
        proba[0] = 1 - probability_of_class
        return proba

    def get_init_raw_predictions(self, X, estimators):
        # https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/ensemble/_gb_losses.py#L645
        probas = estimators.predict_proba(X) # TODO
        probas_of_class = probas[:,1] # esta es la probabilidad de exito
         # log(p / (1 - p)) is the inverse of the sigmoid (expit) function
         # en este caso, probas_of_class deben ser odds
        raw_predictions = np.log(probas_of_class / (1 - probas_of_class))
        return raw_predictions

