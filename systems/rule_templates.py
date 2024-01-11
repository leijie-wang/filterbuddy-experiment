import logging
logger = logging.getLogger(__name__)


class RuleTemplate:
    """Base class for all rule templates."""
    _registry = {}
    schema = {}

    @classmethod
    def register(cls, name, child_cls):
        """Register a child class to the registry."""
        cls._registry[name] = child_cls

    @classmethod
    def get_rule_template(cls, index):
        """Maps a preset index to the corresponding rule template."""
        # Assuming 'index' is a key in the registry
        return cls._registry.get(index)

    @classmethod
    def get_all_schemas(cls):
        """Returns a list of schemas from all registered child classes, also adding the key 'index' to each schema."""
        schemas = []
        for index, child_cls in cls._registry.items():
            schema = child_cls.schema
            schema["index"] = index
            schemas.append(schema)
        return schemas

    @classmethod
    def get_function(cls, input_schema):
        """Class method to return a function based on the input schema.
        This should be overridden by subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @classmethod
    def check_validity(cls, input_schema):
        """Class method to check the validity of the input schema."""
        input_schema["settings"] = {setting["name"]: setting for setting in input_schema["settings"]}
        for setting in cls.schema["settings"]:
            if setting["name"] not in input_schema["settings"]:
                logger.error(f"The input schema should have a setting with the name '{setting['name']}'.")
                return None
        return input_schema

    @classmethod
    def get_term(cls, term_schema):
        words = []
        words.append(term_schema["value"])
        words.extend(term_schema["synonyms"])
        # TODO: we should consider the spelling variants as well
        return words
    
class IncludeWordsRule(RuleTemplate):
    schema = {
        "description": "Catch texts that include some words simultaneously",
        "summary": "Catch texts that include all these words {words} simultaneously",
        "settings": [
            {
                "name": "words",
                "description": "Texts should include each of these words at the same time",
                "type": "list[term]",
            }
        ]
    }

    @classmethod
    def get_function(cls, input_schema):
        """Returns a function that checks if a text includes all the words in the input schema."""
        # check the validity of the input schema by finding the setting with the name words from settings as a list
        input_schema = cls.check_validity(input_schema)
        if input_schema is None:
            return None
        
        setting = input_schema["settings"]["words"]
        words_to_include = []
        for term in setting["value"]:
            words_to_include.extend(cls.get_term(term))

        def _include(self, input):
            for lst in words_to_include:
                if not any(substring in input for substring in lst):
                    return False
            return True
        return _include

class ExcludeWordsRule(RuleTemplate):
    schema = {
        "description": "Catch texts that don't include some words simultaneously",
        "summary": "Catch texts that don't include all these words {words} simultaneously",
        "settings": [
            {
                "name": "words",
                "description": "Texts should not include all these words at the same time",
                "type": "list[term]",
            }
        ]
    }

    @classmethod
    def get_function(cls, input_schema):
        """Returns a function that checks if a text excludes all the words in the input schema."""
        input_schema = cls.check_validity(input_schema)
        if input_schema is None:
            return None
        
        setting = input_schema["settings"]["words"]
        words_to_exclude = []
        for term in setting["value"]:
            words_to_exclude.extend(cls.get_term(term))
        
        def _exclude(self, input):
            for lst in words_to_exclude:
                # if any of the words in a list is not in the input, then return True
                if not any(substring in input for substring in lst):
                    return True
            return False
        return _exclude



class IncludeExcludeWordsRule(RuleTemplate):
    schema = {
        "description": "Texts that include some words but exclude others",
        "summary": "Texts that include these words {include words} but exclude those words {exclude words}",
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

    @classmethod
    def get_function(cls, input_schema):
        """Returns a function that checks if a text includes all the words in the input schema."""
        # check the validity of the input schema
        input_schema = cls.check_validity(input_schema)
        if input_schema is None:
            return None
        
        include_setting = input_schema["settings"]["include words"]
        words_to_include = []
        for term in include_setting["value"]:
            words_to_include.extend(cls.get_term(term))

        exclude_setting = input_schema["settings"]["exclude words"]
        words_to_exclude = []
        for term in exclude_setting["value"]:
            words_to_exclude.extend(cls.get_term(term))

        def _include_exclude(self, input):
            for lst in words_to_include:
                if not any(substring in input for substring in lst):
                    return False
                
            for lst in words_to_exclude:
                if not any(substring in input for substring in lst):
                    return 
            return True

        
    
RuleTemplate.register(0, IncludeWordsRule)
RuleTemplate.register(1, ExcludeWordsRule)
RuleTemplate.register(2, IncludeExcludeWordsRule)