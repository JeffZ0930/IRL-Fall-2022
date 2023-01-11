import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
from sklearn.base import RegressorMixin, ClassifierMixin, is_regressor, is_classifier, clone

from .mobtree import MoBTreeRegressor, MoBTreeClassifier

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

__all__ = ["CustomMobTreeRegressor", "CustomMobTreeClassifier"]


class CustomMobTreeRegressor(MoBTreeRegressor, RegressorMixin):

    def __init__(self, base_estimator, param_dict={}, max_depth=3, min_samples_leaf=50, min_impurity_decrease=0, feature_names=None,
                 split_features=None, n_screen_grid=1, n_feature_search=10, n_split_grid=20, random_state=0, **kargs):

        super(CustomMobTreeRegressor, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 feature_names=feature_names,
                                 split_features=split_features,
                                 n_screen_grid=n_screen_grid,
                                 n_feature_search=n_feature_search,
                                 n_split_grid=n_split_grid,
                                 random_state=random_state)
        self.param_dict = param_dict
        self.base_estimator = base_estimator
        if "random_state" in self.base_estimator.get_params().keys():
            self.base_estimator.set_params(**{"random_state": self.random_state})
        self.base_estimator.set_params(**kargs)

    def build_root(self):

        self.base_estimator.fit(self.x, self.y)
        root_impurity = self.evaluate_estimator(self.base_estimator, self.x, self.y.ravel())
        return root_impurity

    def build_leaf(self, sample_indice):

        if len(self.param_dict) == 0:
            best_estimator = clone(self.base_estimator)
            best_estimator.fit(self.x[sample_indice], self.y[sample_indice].ravel())
        else:
            param_size = 1
            for key, item in self.param_dict.items():
                param_size *= len(item)
            if param_size == 1:
                best_estimator = clone(self.base_estimator)
                best_estimator.set_params(**{key: item[0] for key, item in self.param_dict.items()})
                best_estimator.fit(self.x[sample_indice], self.y[sample_indice].ravel())
            else:
                grid = GridSearchCV(self.base_estimator, param_grid=self.param_dict,
                              scoring={"mse": make_scorer(mean_squared_error, greater_is_better=False)},
                              cv=5, refit="mse", n_jobs=1, error_score=np.nan)
                grid.fit(self.x[sample_indice], self.y[sample_indice].ravel())
                best_estimator = grid.best_estimator_
        predict_func = lambda x: best_estimator.predict(x)
        best_impurity = self.get_loss(self.y[sample_indice], best_estimator.predict(self.x[sample_indice]))
        return predict_func, best_estimator, best_impurity


class CustomMobTreeClassifier(MoBTreeClassifier, RegressorMixin):

    def __init__(self, base_estimator, param_dict={}, max_depth=3, min_samples_leaf=50, min_impurity_decrease=0, feature_names=None,
                 split_features=None, n_screen_grid=1, n_feature_search=10, n_split_grid=20, random_state=0, **kargs):

        super(CustomMobTreeClassifier, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 feature_names=feature_names,
                                 split_features=split_features,
                                 n_screen_grid=n_screen_grid,
                                 n_feature_search=n_feature_search,
                                 n_split_grid=n_split_grid,
                                 random_state=random_state)
        self.param_dict = param_dict
        self.base_estimator = base_estimator
        if "random_state" in self.base_estimator.get_params().keys():
            self.base_estimator.set_params(**{"random_state": self.random_state})
        self.base_estimator.set_params(**kargs)

    def build_root(self):

        self.base_estimator.fit(self.x, self.y)
        root_impurity = self.evaluate_estimator(self.base_estimator, self.x, self.y.ravel())
        return root_impurity

    def build_leaf(self, sample_indice):

        if len(self.param_dict) == 0:
            best_estimator = clone(self.base_estimator)
            best_estimator.fit(self.x[sample_indice], self.y[sample_indice].ravel())
            predict_func = lambda x: best_estimator.decision_function(x)
            best_impurity = self.get_loss(self.y[sample_indice], best_estimator.predict_proba(self.x[sample_indice])[:, 1])
        else:
            param_size = 1
            for key, item in self.param_dict.items():
                param_size *= len(item)
            if param_size == 1:
                best_estimator = clone(self.base_estimator)
                best_estimator.set_params(**{key: item[0] for key, item in self.param_dict.items()})
                best_estimator.fit(self.x[sample_indice], self.y[sample_indice].ravel())
                predict_func = lambda x: best_estimator.decision_function(x)
                best_impurity = self.get_loss(self.y[sample_indice], best_estimator.predict_proba(self.x[sample_indice])[:, 1])
            else:
                if (self.y[sample_indice].std() == 0) | (self.y[sample_indice].sum() < 5) | ((1 - self.y[sample_indice]).sum() < 5):
                    best_estimator = None
                    predict_func = lambda x: np.ones(x.shape[0]) * self.y[sample_indice].mean()
                    best_impurity = self.get_loss(self.y[sample_indice], predict_func(self.x[sample_indice]))
                else:
                    grid = GridSearchCV(self.base_estimator, param_grid=self.param_dict,
                                  scoring={"auc": make_scorer(roc_auc_score, needs_proba=True)},
                                  cv=5, refit="auc", n_jobs=1, error_score=np.nan)
                    grid.fit(self.x[sample_indice], self.y[sample_indice].ravel())
                    best_estimator = grid.best_estimator_

                predict_func = lambda x: best_estimator.decision_function(x)
                best_impurity = self.get_loss(self.y[sample_indice], best_estimator.predict_proba(self.x[sample_indice])[:, 1])
        return predict_func, best_estimator, best_impurity
