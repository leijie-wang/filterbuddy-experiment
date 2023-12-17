class TextFilter:
    def __init__(self, rules):
        self.rules = rules

    def filter_from_rules(self, string_input):
        lowercase_input = string_input.lower()

        # This variable will be True if should be filtered out
        
        rulePrediction = {}

        for item in self.rules:
            should_include_boolean = False
            index = item["index"]
            description = item["description"]
            if index == 0:
                should_include_boolean = self._include(item, lowercase_input)
            elif index == 2:
                should_include_boolean = self._include_exclude(item, lowercase_input)
            rulePrediction[description] = should_include_boolean

        return rulePrediction

    def _include(self, rule, input_text):
        settings = rule["settings"]

        for setting in settings:
            word_to_include = setting["value"]

            # Building a list of all words to look for
            synonyms = setting["synonyms"]
            synonyms.append(word_to_include)

            # Return True if any of the words to include are in the input
            if not any(word in input_text for word in synonyms):
                # If any setting does not meet the condition, return False
                return False

        # If all settings meet the condition, return True
        return True

    def _include_exclude(self, rule, input_text):
        settings = rule["settings"]

        for setting in settings:
            word_to_include = setting[0]["value"]

            # Building a list of all words to look for
            include_synonyms = setting[0]["synonyms"]
            include_synonyms.append(word_to_include)
            word_to_exclude = setting[1]["value"]

            # Building a list of all "not including" words
            exclude_synonyms = setting[1]["synonyms"]
            exclude_synonyms.append(word_to_exclude)

            # Check if any included word is present
            included_word_present = any(include_word in input_text for include_word in include_synonyms)

            excluded_words_present = [exclude_word for exclude_word in exclude_synonyms if exclude_word in input_text]

            # Check if all excluded words are not present
            all_excluded_words_not_present = all(exclude_word not in input_text for exclude_word in exclude_synonyms)

            # Return False if any setting is not 'filtered true'
            if not (included_word_present and all_excluded_words_not_present):
                return False

        # If all settings are 'filtered true', return True
        return True