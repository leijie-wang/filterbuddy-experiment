from sharedsteps.utils import calculate_algorithm_metrics
import logging
import random
import json

SUPPORTED_RULES = [
    {
        "description": "Texts that include some words",
        "summary": "Texts that include all these words {words}",
        "index": 0,
        "settings": [
            {
                "name": "words",
                "description": "Texts should include each of these words at the same time",
                "type": "list[term]",
            }
        ]
    },
    {
        "description": "Texts that exclude some words",
        "summary": "Texts that exclude all these words {words}",
        "index": 1,
        "settings": [
            {
                "name": "words",
                "description": "Texts should exclude each of these words at the same time",
                "type": "list[term]",
            }
        ]
    },
    {
        "description": "Texts that include some words but exclude others",
        "summary": "Texts that include these words {include words} but exclude those words {exclude words}",
        "index": 2,
        "settings": [
            {
                "name": "include words",
                "description": "texts should include each of these words at the same time",
                "type": "list[term]",
            },
            {
                "name": "exclude words",
                "description": "texts should exclude each of these words at the same time",
                "type": "list[term]",
            }
        ]
    }
]

class TreesFilter:

    def __init__(self, rules, debug=False):
        self.debug = debug 
        self.rules = rules
    
    def _test_rule(self, rule, dataset):
        pass

        
    
    def test_model(self, X, y=None):
        """
            There is no training stage for Trees Filter. We only test the model against X, y
            Here, in order to generate explanations for each prediction, 
            instead of building a model with all rules, we generate predictions for each rule individually and aggregate the results for the final prediction.

            @param X: a list of texts
            @param y: a list of 0, 1 representing the labels of the texts

        """
        X_test, y_test = X, y

        
        rule_predictions = [{} for _ in range(len(X_test))]

        for index in range(len(self.rules)):
            rule = self.rules[index]
            rule_id = rule["rule_id"]
            rule_pred = self._test_rule(rule, X_test)
            
            if rule_pred is not None:
                for index in range(len(X_test)):
                    rule_predictions[index][rule_id] = rule_pred[index] if index in rule_pred else None

        prediction = [0 for _ in range(len(X_test))] # overall predictions
        for index in range(len(X_test)):
            rule_pred = rule_predictions[index]
            # aggregate individual predictions using or operation, do not consider None
            valid_prediction = [value for value in rule_pred.values() if value is not None]
            prediction[index] = any(valid_prediction) if len(valid_prediction) > 0 else False
        
        # if the user builds the model interactively, then y_test will be None
        if y_test is not None:
            performance = calculate_algorithm_metrics(y_test, prediction)
        else:
            performance = {}

        return {
            "prediction": prediction,
            "rule_predictions": rule_predictions,
            "performance": performance
        }