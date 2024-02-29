from sharedsteps.utils import calculate_algorithm_metrics
import re
import logging
logger = logging.getLogger(__name__)

ALL_VARIANTS = {
                'a': '[aÁáÀàĂăẮắẰằẴẵẲẳÂâẤấẦầẪẫẨẩǍǎÅåǺǻÄäǞǟÃãȦȧǠǡĄąĄ́ą́Ą̃ą̃ĀāĀ̀ā̀ẢảȀȁA̋a̋ȂȃẠạẶặẬậḀḁȺⱥᶏẚＡａ@4*]',
                'b': '[bb́ḃḅḇƀᵬᶀɓƃⲃбвβƄ♭ɓBB́ḂḄḆɃƁƂⲂБВƅ฿₿ℬʙ*]',
                'c': '[cĆćĈĉČčĊċÇçḈḉȻȼꞒꞓꟄꞔƇƈɕ*]',
                'd': '[dĎďḊḋḐḑD̦d̦ḌḍḒḓḎḏĐđÐðƉɖƊɗᵭᶁᶑȡ*]',
                'e': '[e3ЄєЕеÉéÈèĔĕÊêẾếỀềỄễỂểÊ̄ê̄Ê̌ê̌ĚěËëẼẽĖėĖ́ė́Ė̃ė̃ȨȩḜḝĘęĘ́ę́Ę̃ę̃ĒēḖḗḔḕẺẻȄȅE̋e̋ȆȇẸẹỆệḘḙḚḛɆɇE̩e̩È̩è̩É̩é̩ᶒⱸＥｅ*]',
                'f': '[fḟƒᵮᶂḞƑ*]',
                'g': '[gǴǵĞğĜĝǦǧĠġG̃g̃ĢģḠḡǤǥꞠꞡƓɠᶃＧｇqꝖꝗꝘꝙɋʠ*]',
                'h': '[hĤĥȞȟḦḧḢḣḨḩḤḥḪḫH̱ẖĦħⱧⱨꞪɦΗНн*]',
                'i': '[iÍíi̇́Ììi̇̀ĬĭÎîǏǐÏïḮḯĨĩi̇̃ĮįĮ́į̇́Į̃į̇̃ĪīĪ̀ī̀ỈỉȈȉI̋i̋ȊȋỊịḬḭƗɨᶖİiIıＩｉ1lĺľļḷḹl̃ḽḻłŀƚꝉⱡɫɬꞎᶅɭȴＬｌ*]',
                'j': '[jĵјɟʝʲⅉJĴJ̃Ј*]',
                'k': '[kḰḱǨǩĶķḲḳḴḵƘƙⱩⱪᶄꝀꝁꝂꝃꝄꝅꞢꞣ*]',
                'l': '[lĺḹl̅ḷľłłƚꝉl̥լλ£ℒℓɬɫLĹĽḸL̅ḶŁȽԼΛ£ℒℓɮɫ*]',
                'm': '[mḿṁṃm̀m̂m̄m̌ɱмⰿμḿm̀ɱm̥ᴍᵐᴹMḾṀṂM̀M̂M̄M̌ⱮМⰏΜ*]',
                'n': '[nŃńǸǹŇňÑñṄṅŅņṆṇṊṋṈṉN̈n̈ƝɲŊŋꞐꞑꞤꞥᵰᶇɳȵИиПпＮｎ*]',
                'o': '[ÓóÒòŎŏÔôỐốỒồỖỗỔổǑǒÖöȪȫŐőÕõṌṍṎṏȬȭȮȯO͘o͘ȰȱØøǾǿǪǫǬǭŌōṒṓṐṑỎỏȌȍȎȏƠơỚớỜờỠỡỞởỢợỌọỘộO̩o̩Ò̩ò̩Ó̩ó̩ƟɵꝊꝋꝌꝍⱺＯｏ0*]',
                'p': '[pꝓƥṕṗp̄ꝑⱂπ℗рƿþÞᚦPꝒƤṔṖP̄ꝐⰒΠ¶₱℗ℙΡРᏢǷ*]',
                'q': '[qꝖꝗꝘꝙɋʠＱｑ*]',
                'r': '[rŔŕŘřṘṙŖŗȐȑȒȓṚṛṜṝṞṟR̃r̃ɌɍꞦꞧⱤɽᵲᶉ*]',
                's': '[sŚśṤṥŜŝŠšṦṧṠṡŞşṢṣṨṩȘșS̩s̩ꞨꞩⱾȿʂᶊᵴ*]',
                't': '[tŤťṪṫŢţṬṭȚțṰṱṮṯŦŧȾⱦƬƭƮʈT̈ẗᵵƫȶ*]',
                'u': '[uūúǔùŭûüůųũűȕūṳṵṷṹṻǖǘǜǚủũụưừửữứựυύϋΰὐὔὒὖὑὕὓὗὺῦῠῡῢῧʉʊｕUŪǓÙŬÛÜŮŲŨŰȔṲṴṶṸṺǕǗǙǛỦŨỤƯỪỬỮỨỰԱՄՍɄƱＵ*]',
                'v': '[VṼṾV̇ꝞV̅ѴƲɅＶ*]',
                'w': '[wẃẁŵẘẅẇẉʍʷʬԝｗωѡWẂẀŴẄẆẈԜ₩שШЩѠ*]',
                'x': '[xẍẋᶍx̂x̄x̱хҳӽӿᕽ×✖⨯✗✘☒☓𝄪𝕩XẌẊX̂X̄X̲Χχ͵χΧ✕╳⨉𝕏ㄨメ乂㐅ᚷᚸ*]',
                'y': '[yÝýỲỳŶŷY̊ẙŸÿỸỹẎẏȲȳỶỷỴỵɎɏƳƴỾỿ*]',
                'z': '[zžżẓźẕẑʐʑȥƶⲍʑʐʒƨZŽŻẒẔẐŹƵΖζΖⲌℤ☡乙之Ƨ*]'
            }


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
    
    def _build_word_regex(self, word, variants):
        """
            build a regex for a word
            @param word: a word
        """
        regex = r"\b(?=\w)" # matching the beginning of a word
        for char in word:
            if variants and (char.lower() in ALL_VARIANTS):
                regex += ALL_VARIANTS[char.lower()] + "+"
            else:
                regex += char
        regex += r"\b(?!\w)" # matching the end of a word
        return regex
    
    def _build_statement(self, unit, variants=False):
        regex_list = [self._build_word_regex(word, variants) for word in unit["words"]]
        or_regex = "|".join(regex_list) # a regex that checks if any of the words in the list is in the text
        def func(text):
            match = re.search(or_regex, text, re.IGNORECASE)
            if match:
                return True, match.group()  # Return True and the matched pattern
            else:
                return False, None  # Return False and None if no match
        return func

                
    def _build_rule_function(self, rule):
        units = rule["units"]
        variants = rule["variants"]
        funcs = [(unit["type"], self._build_statement(unit, variants)) for unit in units]
        def rule_function(text):
            patterns = []
            for type, func in funcs:
                result, pattern = func(text)
                if result:
                    patterns.append([type, pattern])

                if type == "include" and not result:
                    return False, patterns
                if type == "exclude" and result:
                    return False, patterns
            return True, patterns
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