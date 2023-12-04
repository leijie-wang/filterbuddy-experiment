from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
import logging
import json, sys, os

logger = logging.getLogger(__name__)

sys.path.append(os.path.join(settings.BASE_DIR, 'datasets'))
BATCH_SIZE = 20

# Create your views here.
def onboarding(request):
    return render(request, 'onboarding.html')

def load_system(request):
    # parse get parameters of the request
    participant_id = request.GET.get('participant_id', default=None)
    condition = request.GET.get('condition', default=None)
    system, task = condition.split('.')
    # parameters = {
    #         "task": task,
    #         "participant_id": participant_id,
    #     }
    logging.debug(f"participant_id: {participant_id}, system: {system}, task: {task}")
    # redirect participants to the assigned system
    if system == "basic-filter":
        return redirect(f'/wordfilter?task={task}&participant_id={participant_id}')
    elif system == "basic-ML":
        return redirect(f'/examplelabel?task={task}&participant_id={participant_id}')
    elif system == "advanced-ML":
        pass
    elif system == "forest-filter":
        pass
    else:
        logging.error("Unknown system: {}".format(system))


def load_data(number, start):
    # read json datasets from dataset folder
    from datasets import toxicity
    mydataset = toxicity.load_dataset(number, start)
    return mydataset

def load_more_data(request):
    last_length = request.GET.get('last_length', default=None)
    if last_length is not None:
        return JsonResponse(load_data(BATCH_SIZE, int(last_length)), safe=False)

def wordfilter(request):
    dataset = load_data(BATCH_SIZE, 0)
    return render(request, 'wordfilter.html', {
        "dataset": json.dumps(dataset),
    })

def examplelabel(request):
    dataset = load_data(BATCH_SIZE, 0)
    return render(request, 'examplelabel.html', {
         "dataset": json.dumps(dataset),
         # use json.dumps to ensure it can be read in js
    })

def trainML(request):
    #todo: train ML model based on user labels in the format of [(text1, 1), (text2, 0), ....]
    pass

def promptwrite(request):
    dataset = load_data(BATCH_SIZE/2, 0)
    return render(request, 'promptwrite.html', {
         "dataset": json.dumps(dataset),
         # use json.dumps to ensure it can be read in js
    })

def ruleconfigure(request):
    dataset = load_data(BATCH_SIZE, 0)
    return render(request, 'ruleconfigure.html', {
         "dataset": json.dumps(dataset),
         # use json.dumps to ensure it can be read in js
    })

# as of now, not sure which function to put this in. will revise with leije when time comes:

def filterFromRules(rules, stringInput):
    # true returns input should be filtered, false returns should not filtered

    lowercaseInput = stringInput.lower()

    # this variable will be true if should be filtered out
    shouldIncludeBoolean = False
    for item in rules:
        description = item["description"]
        if (description == "Texts that include a word"):
            shouldIncludeBoolean = include(item, lowercaseInput)
        elif (description == "Texts that include a word but exclude another word"):
            shouldIncludeBoolean = includeExclude(item, lowercaseInput)
        if (shouldIncludeBoolean == True):
                # as an AND relationship, if any rule comes out to be triggered, then should be filtered out
                return True
    return shouldIncludeBoolean


def include(rule, input):
    settings = rule["settings"]
    wordToInclude = settings[0]["value"]
    # building list of all words to look for 
    synonyms = settings[0]["synonyms"]
    synonyms.append(wordToInclude)

    # Return true if any of the words to include are in the input
    return any(word in input for word in synonyms)

def includeExclude(rule, input):
    settings = rule["settings"]
    wordToInclude = settings[0]["value"]
    
    # building list of all words to look for 
    includeSynonyms = settings[0]["synonyms"]
    includeSynonyms.append(wordToInclude)
    wordToDisclude = settings[1]["value"]

    # building list of all "not including" words
    discludeSynonyms = settings[1]["synonyms"]
    discludeSynonyms.append(wordToDisclude)

    # Check if any included word is present
    included_word_present = any(includeWord in input for includeWord in includeSynonyms)

    # Check if all discluded words are not present
    all_discluded_words_not_present = all(discludeWord not in input for discludeWord in discludeSynonyms)

    # return true if it should be filtered out
    return included_word_present and all_discluded_words_not_present