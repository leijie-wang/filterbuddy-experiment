from sharedsteps.utils import calculate_algorithm_metrics
import stanza
from pattern.en import pluralize, singularize, conjugate, PRESENT, PAST, SG, INFINITIVE, PARTICIPLE, PROGRESSIVE
import re
import logging
logger = logging.getLogger(__name__)

# ALL_VARIANTS = {
#                 'a': '[aÁáÀàĂăẮắẰằẴẵẲẳÂâẤấẦầẪẫẨẩǍǎÅåǺǻÄäǞǟÃãȦȧǠǡĄąĄ́ą́Ą̃ą̃ĀāĀ̀ā̀ẢảȀȁA̋a̋ȂȃẠạẶặẬậḀḁȺⱥᶏẚＡａ@4*]',
#                 'b': '[bb́ḃḅḇƀᵬᶀɓƃⲃбвβƄ♭ɓBB́ḂḄḆɃƁƂⲂБВƅ฿₿ℬʙ*]',
#                 'c': '[cĆćĈĉČčĊċÇçḈḉȻȼꞒꞓꟄꞔƇƈɕ*]',
#                 'd': '[dĎďḊḋḐḑD̦d̦ḌḍḒḓḎḏĐđÐðƉɖƊɗᵭᶁᶑȡ*]',
#                 'e': '[e3ЄєЕеÉéÈèĔĕÊêẾếỀềỄễỂểÊ̄ê̄Ê̌ê̌ĚěËëẼẽĖėĖ́ė́Ė̃ė̃ȨȩḜḝĘęĘ́ę́Ę̃ę̃ĒēḖḗḔḕẺẻȄȅE̋e̋ȆȇẸẹỆệḘḙḚḛɆɇE̩e̩È̩è̩É̩é̩ᶒⱸＥｅ*]',
#                 'f': '[fḟƒᵮᶂḞƑ*]',
#                 'g': '[gǴǵĞğĜĝǦǧĠġG̃g̃ĢģḠḡǤǥꞠꞡƓɠᶃＧｇqꝖꝗꝘꝙɋʠ*]',
#                 'h': '[hĤĥȞȟḦḧḢḣḨḩḤḥḪḫH̱ẖĦħⱧⱨꞪɦΗНн*]',
#                 'i': '[iÍíi̇́Ììi̇̀ĬĭÎîǏǐÏïḮḯĨĩi̇̃ĮįĮ́į̇́Į̃į̇̃ĪīĪ̀ī̀ỈỉȈȉI̋i̋ȊȋỊịḬḭƗɨᶖİiIıＩｉ1lĺľļḷḹl̃ḽḻłŀƚꝉⱡɫɬꞎᶅɭȴＬｌ*]',
#                 'j': '[jĵјɟʝʲⅉJĴJ̃Ј*]',
#                 'k': '[kḰḱǨǩĶķḲḳḴḵƘƙⱩⱪᶄꝀꝁꝂꝃꝄꝅꞢꞣ*]',
#                 'l': '[lĺḹl̅ḷľłłƚꝉl̥լλ£ℒℓɬɫLĹĽḸL̅ḶŁȽԼΛ£ℒℓɮɫ*]',
#                 'm': '[mḿṁṃm̀m̂m̄m̌ɱмⰿμḿm̀ɱm̥ᴍᵐᴹMḾṀṂM̀M̂M̄M̌ⱮМⰏΜ*]',
#                 'n': '[nŃńǸǹŇňÑñṄṅŅņṆṇṊṋṈṉN̈n̈ƝɲŊŋꞐꞑꞤꞥᵰᶇɳȵИиПпＮｎ*]',
#                 'o': '[ÓóÒòŎŏÔôỐốỒồỖỗỔổǑǒÖöȪȫŐőÕõṌṍṎṏȬȭȮȯO͘o͘ȰȱØøǾǿǪǫǬǭŌōṒṓṐṑỎỏȌȍȎȏƠơỚớỜờỠỡỞởỢợỌọỘộO̩o̩Ò̩ò̩Ó̩ó̩ƟɵꝊꝋꝌꝍⱺＯｏ0*]',
#                 'p': '[pꝓƥṕṗp̄ꝑⱂπ℗рƿþÞᚦPꝒƤṔṖP̄ꝐⰒΠ¶₱℗ℙΡРᏢǷ*]',
#                 'q': '[qꝖꝗꝘꝙɋʠＱｑ*]',
#                 'r': '[rŔŕŘřṘṙŖŗȐȑȒȓṚṛṜṝṞṟR̃r̃ɌɍꞦꞧⱤɽᵲᶉ*]',
#                 's': '[sŚśṤṥŜŝŠšṦṧṠṡŞşṢṣṨṩȘșS̩s̩ꞨꞩⱾȿʂᶊᵴ*]',
#                 't': '[tŤťṪṫŢţṬṭȚțṰṱṮṯŦŧȾⱦƬƭƮʈT̈ẗᵵƫȶ*]',
#                 'u': '[uūúǔùŭûüůųũűȕūṳṵṷṹṻǖǘǜǚủũụưừửữứựυύϋΰὐὔὒὖὑὕὓὗὺῦῠῡῢῧʉʊｕUŪǓÙŬÛÜŮŲŨŰȔṲṴṶṸṺǕǗǙǛỦŨỤƯỪỬỮỨỰԱՄՍɄƱＵ*]',
#                 'v': '[VṼṾV̇ꝞV̅ѴƲɅＶ*]',
#                 'w': '[wẃẁŵẘẅẇẉʍʷʬԝｗωѡWẂẀŴẄẆẈԜ₩שШЩѠ*]',
#                 'x': '[xẍẋᶍx̂x̄x̱хҳӽӿᕽ×✖⨯✗✘☒☓𝄪𝕩XẌẊX̂X̄X̲Χχ͵χΧ✕╳⨉𝕏ㄨメ乂㐅ᚷᚸ*]',
#                 'y': '[yÝýỲỳŶŷY̊ẙŸÿỸỹẎẏȲȳỶỷỴỵɎɏƳƴỾỿ*]',
#                 'z': '[zžżẓźẕẑʐʑȥƶⲍʑʐʒƨZŽŻẒẔẐŹƵΖζΖⲌℤ☡乙之Ƨ*]'
#             }

ALL_VARIANTS = {
                'a': '[a@4*]',
                'b': '[b(13)8*]',
                'c': '[c*]',
                'd': '[d*]',
                'e': '[e3Єє*]',
                'f': '[f*]',
                'g': '[g&6*]',
                'h': '[h#*]',
                'i': '[i!1*]',
                'j': '[j*]',
                'k': '[k*]',
                'l': '[l1*]',
                'm': '[m*]',
                'n': '[n*]',
                'o': '[o0*]',
                'p': '[p*]',
                'q': '[q9*]',
                'r': '[r(12)*]',
                's': '[s5$*]',
                't': '[t*]',
                'u': '[u*]',
                'v': '[V*]',
                'w': '[w*]',
                'x': '[x*]',
                'y': '[y*]',
                'z': '[z2*]'
            }

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')

class TreesFilter:

    @classmethod
    def train(cls, system, **kwargs):
        rules = system.read_rules()
        if len(rules) == 0:
            return False, "No rules found for the participant"
        
        tree_filter = TreesFilter(rules)
        return True, tree_filter

    def __init__(self, rules, debug=False):
        self.debug = debug 
        self.rules = rules
        # sorted by priority from small to large
        self.rules = sorted(self.rules, key=lambda x: x["priority"])
        self.relevant_punctuation = "!?." 

        """
            To resolve the StopIteration error in the pattern library, 
            we need to call the conjugate function once before the first call in the _get_word_variants function to load the pattern library
            See this issue https://github.com/piskvorky/gensim/issues/2438
        """
        try:
            conjugate("try", tense=PAST)
        except Exception as e:
            print(f"Error in loading the pattern library: {e}")
    
    def escape_regex(self, char):
        if char in "^$.*+?{}[]|()\\":  # List of special characters that need to be escaped
            return "\\" + char
        return char
    
    def _get_word_variants(self, word):
        """
            for a noun, return its singular and plural forms
            for a verb, return its base form, present third person, past, present participle, past participle
            note that the pos algorithm is not perfect, so we may not always get what we want, for instance, catching is considered a noun but not a verb
        """
        # determine if it is actuall a word as opposed to a string of several words, or punctuation
        if len(word.split()) > 1 or not word.isalpha():
            return [word]

        doc = nlp(word)
        word_data = doc.sentences[0].words[0]
        pos_tag = word_data.xpos
        base_form = word_data.lemma
        # print(f"word: {word}, pos_tag: {pos_tag}, base_form: {base_form}")
        word_list = [word]
        if pos_tag.startswith('VB'):
            # Generate and return other tenses of the verb
            
            present_third_person = conjugate(base_form, tense=PRESENT, person=3, number=SG)  # He/She/It form
            past = conjugate(base_form, tense=PAST)  # Simple past
            present_participle = conjugate(base_form, tense=PARTICIPLE, aspect='progressive')  # -ing form
            past_participle = conjugate(base_form, tense=PAST, aspect='perfective')  # Perfect aspect
            word_list.extend([base_form, present_third_person, past, present_participle, past_participle])
        elif pos_tag.startswith('NN'):
            if pos_tag == 'NNS':
                word_list.append(singularize(word))
            else:
                word_list.append(pluralize(base_form))
        word_list = [word for word in list(set(word_list)) if word is not None]
        # print(f"word_list: {word_list} for word: {word}")
        return word_list
    
    def _build_word_regex(self, word, variants):
        """
            build a regex for a word
            @param word: a word
        """
        if all(char in self.relevant_punctuation for char in word):
            # Escape each character (since some punctuation marks are special in regex) and join them
            regex = "".join(self.escape_regex(char) for char in word)
            # Wrap the escaped sequence in a regex that matches one or more occurrences
            regex = f"({regex})+"
            # logger.info(f"regex: {regex} for word: {word}")
            return regex
        else:
            regex = r"\b(?=\w)" # matching the beginning of a word
            for char in word:
                if variants and (char.lower() in ALL_VARIANTS):
                    regex += ALL_VARIANTS[char.lower()] + "+"
                else:
                    regex += self.escape_regex(char)
            regex += r"(?=\W(?!_)|$)" # matching the end of a word
            # logger.info(f"regex: {regex} for word: {word}")
            return regex
    
    def _build_statement(self, unit, variants=False):
        word_list = [self._get_word_variants(word) for word in unit["words"]]
        word_list = list(set([word for sublist in word_list for word in sublist])) # flatten the list and remove duplicates
        regex_list = [self._build_word_regex(word, variants) for word in word_list]
        or_regex = "|".join(regex_list) # a regex that checks if any of the words in the list is in the text
        def func(text):
            try:
                match = re.search(or_regex, text, re.IGNORECASE if variants else 0)
                if match:
                    return True, match.group()  # Return True and the matched pattern
                else:
                    return False, None  # Return False and None if no match
            except Exception as e:
                logger.error(f"Error in matching the text: {text} with the regex: {or_regex}. Error: {e}")
                return False, None
        return func

                
    def _build_rule_function(self, rule):
        units = rule["units"]
        variants = rule["variants"]
        funcs = [(unit["type"], self._build_statement(unit, variants)) for unit in units]
        def rule_function(text):
            patterns = []
            final_result = True
            for type, func in funcs:
                result, pattern = func(text)
                patterns.append([type, pattern])

                if type == "include" and not result:
                    final_result = False
                if type == "exclude" and result:
                    final_result = False
            return final_result, patterns
        return rule_function

    def _test_rule(self, rule, dataset):
        rule_function = self._build_rule_function(rule)
        predictions = {}
        for index in range(len(dataset)):
            pred, patterns = rule_function(dataset[index])
            predictions[index] = {
                "pred": pred,
                "patterns": patterns
            }
        return predictions

    def test_model(self, X, y=None):
        """
            There is no training stage for Trees Filter. We only test the model against X, y
            Here, in order to generate explanations for each prediction, 
            instead of building a model with all rules, we generate predictions for each rule individually and aggregate the results for the final prediction.

            @param X: a list of texts
            @param y: a list of 0, 1 representing the labels of the texts

        """
        X_test, y_test = X, y

        
        texts_predictions = [[] for _ in range(len(X_test))]

        for index in range(len(self.rules)):
            rule = self.rules[index]
            rule_id = rule["id"]
            rule_pred = self._test_rule(rule, X_test)
            
            if rule_pred is not None:
                for index in range(len(X_test)):
                    # the difference between None and 0/1 is still important for the frontend to display, even though None predictions are treated as 0 in the classifier
                    texts_predictions[index].append({
                        "id": rule_id,
                        "prediction": rule["action"] if rule_pred[index]["pred"] else None,
                        "patterns": rule_pred[index]["patterns"]
                    })
            
        prediction = [None for _ in range(len(X_test))] # overall predictions
        for index in range(len(X_test)):
            text_pred = texts_predictions[index]
            """ 
                we have already sorted the rules by priority in the constructor
                use the action of the first rule has a True prediction and the highest priority as the final prediction
            """
            for pred in text_pred:
                if pred["prediction"] is not None:
                    prediction[index] = pred["prediction"]
                    break
            
        # we still make sure that the backend operates with a priority framework but we return the result in a simplified way.
        prediction = [(0 if pred is None else pred) for pred in prediction]  
        
        # if the user builds the model interactively, then y_test will be None
        if y_test is not None:
            # we treat None (not affected texts) as approved texts, which is 0
            performance = calculate_algorithm_metrics(y_test, prediction)
        else:
            performance = {}

        return {
            "prediction": prediction,
            "texts_predictions": texts_predictions,
            "performance": performance
        }