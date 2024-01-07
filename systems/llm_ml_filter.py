from  systems.llm_filter import LLMFilter
from systems.ml_filter import MLFilter

class LLM_ML_MixedFilter:

    def __init__(self, prompts, model_name):
       """
            model_name: the name of the model that will actually be trained for filtering
            prompts: a list of prompts that will be used for LLM labeling
       """
       self.model_name = model_name
       self.prompts = prompts
       self.ml_filter = None

    def train_model(self, X):
        """
            first label the texts using LLM, then train a model using the labels
            @param X: a list of texts
        """
        X_train = X

        llm_filter = LLMFilter(self.prompts, debug=False)
        print(f"starting labeling {len(X_train)} examples using LLM.")

        y_train = llm_filter.test_model(X=X_train, y=None)["prediction"]
        
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


       