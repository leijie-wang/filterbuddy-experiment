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


class IncludeWordsRule(RuleTemplate):
    schema = {
        "description": "Texts that include some words",
        "summary": "Texts that include all these words {words}",
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
        settings = [setting for setting in input_schema["settings"] if setting["name"] == "words"]
        if len(settings) != 1:
            logger.error("The input schema should have exactly one setting with the name 'words'.")
            return None
        setting = settings[0]

        words_to_include = []
        for term in setting["value"]:
            words = []
            words.append(term["value"])
            words.extend(term["synonyms"])
            # TODO: we should consider the spelling variants as well
            words_to_include.append(words)

        def _include(self, input):
            for lst in words_to_include:
                if not any(substring in input for substring in lst):
                    return False
            return True
        return _include

class ExcludeWordsRule(RuleTemplate):
    schema = {
        "description": "Texts that exclude some words",
        "summary": "Texts that exclude all these words {words}",
        "settings": [
            {
                "name": "words",
                "description": "Texts should exclude each of these words at the same time",
                "type": "list[term]",
            }
        ]
    }

    @classmethod
    def get_function(cls, input_schema):
        """Returns a function that checks if a text excludes all the words in the input schema."""
        # check the validity of the input schema

        def check_exclude_words(text):
            return all(word not in text for word in input_schema["settings"])
        return check_exclude_words
    
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

        def check_include_exclude_words(text):
            return all(word in text for word in input_schema["settings"]["include words"]) and all(word not in text for word in input_schema["settings"]["exclude words"])
        return check_include_exclude_words
    
RuleTemplate.register(0, IncludeWordsRule)
RuleTemplate.register(1, ExcludeWordsRule)
RuleTemplate.register(2, IncludeExcludeWordsRule)