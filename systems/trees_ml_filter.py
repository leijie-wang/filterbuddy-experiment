from  systems.trees_filter import TreesFilter
from systems.ml_filter import MLFilter

class Trees_ML_MixedFilter:

    @classmethod
    def train(cls, participant_id, **kwargs):
        from sharedsteps.utils import read_rules_from_database

        dataset = kwargs["dataset"]
        stage = kwargs["stage"]
        
        rules = read_rules_from_database(participant_id, stage)
        if len(rules) == 0:
            return False,"No rules found for the participant"
        
        trees_ml_filter = Trees_ML_MixedFilter(rules, "Bayes")
        
        X_train = dataset.load_train_dataset(participant_id, size=360)
        X_train = [item["text"] for item in X_train]
        
        trees_ml_filter.train_model(X=X_train)
        return True, trees_ml_filter
    
    def __init__(self, rules, model_name):
       """
            model_name: the name of the model that will actually be trained for filtering
            rules: a list of rules that will be used for tree labeling
       """
       self.model_name = model_name
       self.rules = rules
       self.ml_filter = None

    def train_model(self, X):
        """
            first label the texts using LLM, then train a model using the labels
            @param X: a list of texts
        """
        X_train = X

        tree_filter = TreesFilter(self.rules)
        print(f"starting labeling {len(X_train)} examples using trees.")

        y_train = tree_filter.test_model(X=X_train, y=None)["prediction"]
        # remember to convert None to 0 (approved by default)
        y_train = [(0 if pred is None else pred) for pred in y_train]

        
        self.ml_filter = MLFilter(self.model_name)
        train_results = self.ml_filter.train_model(X=X_train, y=y_train)
        return train_results
    
    def test_model(self, X, y):
        """
            Test the model against X, y
        """

        if self.ml_filter is None:
            print('Classifier not trained or stale! Please retrain via .train_model()')
            return
        
        X_test, y_test = X, y
        return self.ml_filter.test_model(X=X_test, y=y_test)


       