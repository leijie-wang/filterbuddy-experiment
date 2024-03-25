from django.shortcuts import render, redirect
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from datasets.dataset import BATCH_SIZE, Dataset, TEST_SIZE
import sharedsteps.utils as utils
import logging
import json, random
from sharedsteps.models import SYSTEMS, Participant
from systems.ml_filter import MLFilter


logger = logging.getLogger(__name__)


BuildDataSet = Dataset("old.csv")
UpdateDataSet = Dataset("new.csv")
TutorialDataset = Dataset("tutorial.csv")


SurveyDict = {
    SYSTEMS.EXAMPLES_ML.value: "https://docs.google.com/forms/d/e/1FAIpQLSeUp58HnYoXvu2-kbsJgqMYOm4eJMfwi-2HEbc__KkPtRtNvQ/viewform?embedded=true",
    SYSTEMS.RULES_TREES.value: "https://docs.google.com/forms/d/e/1FAIpQLSd3Lxs1QFkA1CZekWv9lyeFLNVagzbNZkIja54kJ_FVLHKuxA/viewform?embedded=true",
    SYSTEMS.PROMPTS_LLM.value: "https://docs.google.com/forms/d/e/1FAIpQLSe7ugiUFFx9ix3czO8P49oVXmhd9sINgaMcTEestWLX10-2HQ/viewform?embedded=true",
    "FinalSurvey": "https://docs.google.com/forms/d/e/1FAIpQLSetHVEbgCSLB-XpityPUaCLX6y1p959hO8NwUJwSo_AnV0VtA/viewform?embedded=true"
}

def onboarding(request):
    """
        the starting point of the experiment
    """
    participant_id = request.GET.get('participant_id', default=None)
    group = request.GET.get('group', default=None)
    restart = "restart" in request.GET
    debug = request.GET.get('debug', default="false") == "true"

    participant = None
    if participant_id is None:
        participant = Participant.create_participant(group)
        logger.info(f"participant {participant.participant_id} is created")
    else:
        participant = Participant.objects.get(participant_id=participant_id)
        stage = request.GET.get('stage', default=None)
        system = request.GET.get('system', default=None)
        if stage is not None and system is not None:
            participant.update_progress(stage, system)

    if restart:
        participant.progress = 1
        participant.save() 

    progress = participant.progress
    return render(request, 'new_onboarding.html', {
        "participant_id": participant.participant_id,
        "group": participant.group,
        "progress": progress,
        "debug": debug
    })
    # return redirect(f"/groundtruth/?participant_id={participant.participant_id}&stage={participant.stage}")

def label_ground_truth(request):
    """
        for both build and update stages, the participant will label the ground truth as the first step
    """
    participant_id = request.GET.get('participant_id', default=None)
    debug = request.GET.get('debug', default="false") == "true"

    status, error_message = utils.check_parameters(participant_id)
    if not status:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    participant = Participant.objects.get(participant_id=participant_id)
    stage, _ = participant.get_stage_system()
    
    if stage == "build":
        dataset = BuildDataSet.load_test_dataset(participant_id)
    elif stage == "update":
        dataset = UpdateDataSet.load_test_dataset(participant_id)
        
    condition = participant.get_condition(SYSTEMS.EXAMPLES_ML.value)
    if condition:
        labeled_examples = condition.get_groundtruth_dataset(stage)
        if len(labeled_examples) != 0:
            logger.warning(f"Participant has only labeled {len(labeled_examples)} examples out of {TEST_SIZE}")
            labeled_ids = [item["datum_id"] for item in labeled_examples]
            for datum in dataset:
                if datum["datum_id"] in labeled_ids:
                    datum["label"] = labeled_examples[labeled_ids.index(datum["datum_id"])]["label"]
            
        

    return render(request, 'groundtruth.html', {
            "dataset": json.dumps(dataset),
            "participant_id": participant_id,
            "stage": stage,
            "debug": debug
        })

def store_groundtruth(request):
    """
        store the ground truth labeled by the participant at the stage of build or update
    """
    request_data = json.loads(request.body)
    participant_id = request_data.get('participant_id')
    stage = request_data.get('stage')
    dataset = request_data.get('dataset')
    logs = request_data.get('logs')

    # utils.save_logs(participant_id, stage, logs) // we don't save logs for the ground truth labeling
    status, error_message = utils.check_parameters(participant_id, stage)
    if not status:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    from sharedsteps.models import GroundTruth
    participant = Participant.objects.get(participant_id=participant_id)
    GroundTruth.save_groundtruth(participant, stage, dataset)
    return JsonResponse({"status": True, "message": "Participants' ground truth are stored successfully"}, safe=False)

def load_system(request):
    participant_id = request.GET.get('participant_id', default=None)
    tutorial = request.GET.get('tutorial', default="false") 
    debug = request.GET.get('debug', default="false") == "true"

    status, error_message = utils.check_parameters(participant_id)
    if not status:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    participant = Participant.objects.get(participant_id=participant_id)
    stage, system = participant.get_stage_system()

    debug_parameter = f"&debug=true" if debug else ""
    if system == SYSTEMS.EXAMPLES_ML.value:
        return redirect(f'/examplelabel/?participant_id={participant_id}&system={system}&stage={stage}&tutorial={tutorial}' + debug_parameter)
    elif system == SYSTEMS.RULES_TREES.value:
        return redirect(f'/ruleconfigure/?participant_id={participant_id}&system={system}&stage={stage}&tutorial={tutorial}' + debug_parameter)
    elif system == SYSTEMS.PROMPTS_LLM.value:
        return redirect(f'/promptwrite/?participant_id={participant_id}&system={system}&stage={stage}&tutorial={tutorial}' + debug_parameter)
    else:
        logging.error("System unsupported yet: {}".format(system))

def examplelabel(request):
    participant_id = request.GET.get('participant_id', default=None)
    stage = request.GET.get('stage', default="build")
    system_name= request.GET.get('system')
    tutorial = request.GET.get('tutorial', default="false") == "true"
    debug = request.GET.get('debug', default="false") == "true"

    status, error_message = utils.check_parameters(participant_id, stage, system_name)
    if not status:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    dataset = []
    excluded_ids = []
    if tutorial:
        dataset = TutorialDataset.load_train_dataset(participant_id, size=BATCH_SIZE)
        # there is not test dataset for the tutorial stage
        excluded_ids = TutorialDataset.get_excluded_ids(participant_id, size=BATCH_SIZE, test_include=False)
    elif stage == "build":
        dataset = BuildDataSet.load_train_dataset(participant_id, size=BATCH_SIZE)
        excluded_ids = BuildDataSet.get_excluded_ids(participant_id, size=BATCH_SIZE)
    elif stage == "update":
        dataset = UpdateDataSet.load_train_dataset(participant_id, size=BATCH_SIZE)
        excluded_ids = UpdateDataSet.get_excluded_ids(participant_id, size=BATCH_SIZE)

    time_spent = 0
    if not tutorial:
        participant = Participant.objects.get(participant_id=participant_id)
        condition = participant.get_condition(SYSTEMS.EXAMPLES_ML.value)
        system = condition.get_latest_system(stage=stage)
        time_spent = condition.get_time_spent(stage)
        if system:
            labeled_examples = system.read_examples()
            labeled_ids = [item["datum_id"] for item in labeled_examples]
            logger.warning(f"Participant ${participant_id} has already labeled {len(labeled_examples)} examples")
            for datum in dataset:
                if datum["datum_id"] not in labeled_ids:
                    labeled_examples.append(datum)
                    labeled_ids.append(datum["datum_id"])
            dataset = labeled_examples
            excluded_ids = labeled_ids
        

    return render(request, 'examplelabel.html', {
            "dataset": json.dumps(dataset),
            "excluded_ids": json.dumps(excluded_ids),
            "participant_id": participant_id,
            "stage": stage,
            "system": system_name,
            "time_spent": time_spent,
            "tutorial": tutorial,
            "debug": debug
        })

def active_learning(request):
    request_data = json.loads(request.body)
    
    stage = request_data.get('stage')
    dataset = request_data.get('dataset')
    excluded_ids = request_data.get('excluded_ids')
    active_learning = request_data.get('active_learning')
    tutorial = request_data.get('tutorial')

    dataset_left = []
    if tutorial:
        dataset_left = TutorialDataset.load_data_from_ids(excluded_ids=excluded_ids)
    elif stage == "build":
        dataset_left = BuildDataSet.load_data_from_ids(excluded_ids=excluded_ids)
    elif stage == "update":
        dataset_left = UpdateDataSet.load_data_from_ids(excluded_ids=excluded_ids)
    
    if not tutorial and active_learning:
        X_train = [item["text"] for item in dataset]
        y_train = [item["label"] for item in dataset]

        logger.info(f"training a model with {len(X_train)} examples labeled by the participant on a total of {len(dataset_left)} examples left")
        ml_filter = MLFilter("Bayes")
        ml_filter.train_model(X=X_train, y=y_train)

        X_predict = [item["text"] for item in dataset_left]
        y_prob = ml_filter.predict_prob(X=X_predict)
        uncertainties = [ 1 - abs(prob[0] - prob[1]) for prob in y_prob]
        # get the indices of the most uncertain samples
        selected_indices = sorted(range(len(uncertainties)), key=lambda i: uncertainties[i], reverse=True)[:BATCH_SIZE]
        logger.info(f"selected {BATCH_SIZE} samples with the uncertainties greater than {uncertainties[selected_indices[-1]]}")
        next_batch = [dataset_left[i].copy() for i in selected_indices]
        for index in range(len(selected_indices)):
            data_index = selected_indices[index]
            next_batch[index]["suggested_label"] = int(y_prob[data_index][1] > 0.5)
    else:
        # randomly select the next batch
        next_batch = random.sample(dataset_left, BATCH_SIZE)

    return JsonResponse(
        {
            "status": True,
            "message": "Successfully selected the next batch",
            "data": {
                "next_batch": next_batch
            }}, 
        safe=False
    )

def store_labels(request):
    """
        request.body
        {
            "dataset": [
                {
                    "text": "text1",
                    "label": 1
                },
                {
                    "text": "text2",
                    "label": 0
                }
            ],
            "participant_id": "123456"
            "stage": "build" or "update",
            "logs": [...]
        }
    """
    request_data = json.loads(request.body)
    stage = request_data.get('stage')
    participant_id = request_data.get('participant_id')

    status, error_message = utils.check_parameters(participant_id, stage)
    if not status:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    participant = Participant.objects.get(participant_id=participant_id)
    condition = participant.get_condition(SYSTEMS.EXAMPLES_ML.value)

    logs = request_data.get('logs', None)
    if logs:
        condition.save_logs(logs)

    
    time_spent = request_data.get('time_spent')
    system = condition.create_system(time_spent, stage)
    system.save_system(dataset=request_data.get('dataset'))
    return JsonResponse(
                    {
                        "status": True, 
                        "message": "Participants' labeled examples are stored successfully"
                    }, 
                safe=False
            )

def promptwrite(request):
    participant_id = request.GET.get('participant_id', default=None)
    stage = request.GET.get('stage', default="build")
    system_name = request.GET.get('system')
    tutorial = request.GET.get('tutorial', default="false") == "true"
    debug = request.GET.get('debug', default="false") == "true"

    status, error_message = utils.check_parameters(participant_id, stage, system_name)
    if not status:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    dataset = []
    if tutorial:
        dataset = TutorialDataset.load_train_dataset(participant_id)
    elif stage == "build":
        dataset = BuildDataSet.load_train_dataset(participant_id)
    elif stage == "update":
        dataset = UpdateDataSet.load_train_dataset(participant_id)

    prompts = []
    time_spent = 0
    if not tutorial:
        participant = Participant.objects.get(participant_id=participant_id)
        condition = participant.get_condition(SYSTEMS.PROMPTS_LLM.value)
        system = condition.get_latest_system(stage=stage)
        time_spent = condition.get_time_spent(stage)
        if system:
            prompts = system.read_prompts()

    return render(request, 'promptwrite.html', {
        "participant_id": participant_id,
        "stage": stage,
        "system": system_name,
        "dataset": json.dumps(dataset),
        "instructions": json.dumps(prompts),
        "time_spent": time_spent,
        "tutorial": tutorial,
        "debug": debug
    })

def store_prompts(request):
    """
        request.body
        {
            "prompts": [
                {
                    "rubric": "Catch all texts that involve race-related hate speech",
                    "positives": ["You are just some mad browskin because Europeans styled on your ass"],
                    "negatives": ["Your daily reminder that the vast majority of athletes are hilariously stupid"],
                },
            ],
            "participant_id": "123456"
        }
    """
    request_data = json.loads(request.body)
    participant_id = request_data.get('participant_id')
    stage = request_data.get("stage")
    
    status, error_message = utils.check_parameters(participant_id, stage)
    if not status:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    participant = Participant.objects.get(participant_id=participant_id)
    condition = participant.get_condition(SYSTEMS.PROMPTS_LLM.value)

    logs = request_data.get('logs', None)
    if logs:
        condition.save_logs(logs)

    
    time_spent = request_data.get('time_spent')
    system = condition.create_system(time_spent, stage)
    system.save_system(prompts=request_data.get("instructions"))
    return JsonResponse(
                {
                    "status": True, 
                    "message": "Participants' prompts are stored successfully"
                }, 
                safe=False
            )

def ruleconfigure(request):
    participant_id = request.GET.get('participant_id', default=None)
    stage = request.GET.get('stage', default="build")
    system_name = request.GET.get('system')
    tutorial = request.GET.get('tutorial', default="false") == "true"
    debug = request.GET.get('debug', default="false") == "true"

    status, error_message = utils.check_parameters(participant_id, stage, system_name)
    if not status:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    dataset = []
    if tutorial:
        dataset = TutorialDataset.load_train_dataset(participant_id)
    elif stage == "build":
        dataset = BuildDataSet.load_train_dataset(participant_id)
    elif stage == "update":
        dataset = UpdateDataSet.load_train_dataset(participant_id)

    rules = []
    time_spent = 0
    if not tutorial:
        participant = Participant.objects.get(participant_id=participant_id)
        condition = participant.get_condition(SYSTEMS.RULES_TREES.value)
        system = condition.get_latest_system(stage=stage)
        time_spent = condition.get_time_spent(stage)
        if system:
            rules = system.read_rules()


    return render(request, 'ruleconfigure.html', {
            "participant_id": participant_id,
            "stage": stage,
            "system": system_name,
            "dataset": json.dumps(dataset),
            "instructions": json.dumps(rules),
            "time_spent": time_spent,
            "tutorial": tutorial,
            "debug": debug
        })

def store_rules(request):
    """
    request.body
    {
        "rules": [
            {
                "name": "Catch all texts that involve race-related hate speech",
                "action": 1 for "remove" and 0 for "approve",
                "units": [
                    {
                        "type": "include" or "exclude",
                        "words": list[str]
                    }
                ]
            },
        ],
        "participant_id": "123456",
        "stage": "build" or "update",
        "logs": [...]
    }
    """
    request_data = json.loads(request.body)
    participant_id = request_data.get('participant_id')
    stage = request_data.get('stage')
    
    status, error_message = utils.check_parameters(participant_id, stage)
    if not status:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    participant = Participant.objects.get(participant_id=participant_id)
    condition = participant.get_condition(SYSTEMS.RULES_TREES.value)

    logs = request_data.get('logs', None)
    if logs:
        condition.save_logs(logs)

    
    time_spent = request_data.get('time_spent')
    system = condition.create_system(time_spent, stage)
    system.save_system(rules=request_data.get("instructions"))
    return JsonResponse(
        {
            "status": True,
            "message": f"Participants' {participant_id} rules are stored successfully for {stage}"
        },
        safe=False
    )

def get_similar_phrases(request):
    request_data = json.loads(request.body)
    phrases = request_data.get("phrases")
    chatbot = utils.ChatCompletion()
        
    response = chatbot.chat_completion(
        system_prompt="""
            You are expected to suggest 5 the most similar phrases to a given list of phrases. 
            You should first consider different tenses and forms of a word (such as plural or singular forms for a noun; past/participle tense for a verb), then synonyms of both words and phrases, and then finally commonly used typos of the given phrases.
            Only return phrases that are common in everyday users’ social media comments. 
            RETURN YOUR RESULTS in the JSON format {"results": [a list of phrases]}

            Example:
            1) asshole, stupid, idiot --> idiots, stupidity, assholes
            2) fuck --> f**k, fucking motherfucker
            3) kill, murder --> kills, killed, killing
            4) redneck, Trump, MAGA --> right-wing, Republican, alt-right
        """,
        user_prompt=f"Given the following phrases: {', '.join(phrases)}",
    )
    logger.info(response)
    response = json.loads(response or "{}")
    if "results" in response:
        logger.info(response["results"])
        return JsonResponse(
            {
                "status": True,
                "message": "Successfully got the similar phrases",
                "data": {
                    "phrases": response["results"]
                }
            },
            safe=False
        )

def get_rephrase_instruction(request):
    request_data = json.loads(request.body)
    instruction = request_data.get("instruction")
    positives = request_data.get("positives")
    negatives = request_data.get("negatives")

    chatbot = utils.ChatCompletion()
    user_prompt = f"Given the following prompt: {instruction}"
    if len(positives) > 0:
        user_prompt += "\nExamples users want to remove with this prompt:" +  '\n\t'.join(positives[:2])
    if len(negatives) > 0:
        user_prompt += "\nExamples users do not want to remove with this prompt:" +  '\n\t'.join(negatives[:2])
    
    response = chatbot.chat_completion(
        system_prompt="""
            Users are writing prompts to decide which content they want to keep or remove from their social media. You are expected to improve their prompts. 
            Specifically, you should find out ambiguous concepts or phrases and add annotations. You should only make changes under 10 words.
            Users will also sometimes provide a list of positive examples and negative examples to help you understand the context.

            ##### Examples
            1. Catch all comments that promote hate speech ==> Remove comments that promote hate speech targeting races, genders, or religions
            2. Delete all comments that claim that we should not impose gun control policies ==> Remove comments that reject gun control policies without any supporting argument.
            3. Do not keep comments that directly insult a specific person using words like insane, stupid, morons ==> Remove comments that directly insult a specific person,  using derogatory words such as "insane," "stupid," or "morons."

            RETURN YOUR RESULTS in the JSON format {"results": "the improved prompt"}. REMEMBER to keep your changes under 10 words.
        """,
        user_prompt= user_prompt
    )

    response = json.loads(response or "{}")
    if "results" in response:
        logger.info(response["results"])
        return JsonResponse(
            {
                "status": True,
                "message": "Successfully rephrased the prompt",
                "data": {
                    "instruction": response["results"]
                }
            },
            safe=False
        )

def validate_page(request):
    
    # parse out the participant id fr dom the request GET parameters
    participant_id = request.GET.get('participant_id', default=None)
    system_name = request.GET.get('system', default=None)
    stage = request.GET.get('stage', default=None)
    debug = request.GET.get('debug', default="false") == "true"

    status, error_message = utils.check_parameters(participant_id, stage, system_name)
    if not status:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    participant = Participant.objects.get(participant_id=participant_id)
    condition = participant.get_condition(system_name)
    test_dataset = condition.get_groundtruth_dataset(stage)

    status, response = test_system(condition, stage, test_dataset)
    if status == 0:
        return HttpResponse(f"Error: {response}")
    elif status == 1:
        # for the LLM filter, we set up a job in the backend
        task_id = response
        if stage == "update":
            _, old_task_id = test_system(condition, "build", test_dataset)
        else:
            old_task_id = ""

        return render(request, 'validate.html', {
            "dataset": json.dumps(test_dataset),
            "participant_id": participant_id,
            "system": system_name,
            "stage": stage,
            "old_task_id": old_task_id,
            "task_id": task_id,
            "debug": debug
        })
    else:
        test_results = response
        condition.save_test_results(stage, test_results["prediction"])

        # we want to show the performance of the old system on the new dataset
        old_test_results = {}
        if stage == "update":
            status, old_test_results = test_system(condition, "build", test_dataset)
            condition.save_test_results("update", old_test_results["prediction"], old=True)

        return render(request, 'validate.html', {
            "dataset": json.dumps(test_dataset),
            "participant_id": participant_id,
            "system": system_name,
            "stage": stage,
            "test_results": json.dumps(test_results),
            "old_test_results": json.dumps(old_test_results),
            "task_id": "",
            "debug": debug
        })
        
def test_filter(request):
    request_data = json.loads(request.body)
    dataset = request_data.get('dataset')
    participant_id = request_data.get('participant_id')
    stage = request_data.get('stage')
    if stage == "update":
        status, response = test_system(participant_id, "build", dataset)
        old_performance = utils.calculate_stage_performance(participant_id, "build")
        if status == 0:
            return JsonResponse({"status": False, "message": response}, safe=False)
        elif status == 1:
            task_id = response
            return JsonResponse({
                "status": True,
                "message": "Successfully started the testing",
                "data": {
                    "old_performance": old_performance,
                    "task_id": task_id
                }
            })
        else:
            test_results = response
            utils.save_test_results(participant_id, stage, test_results["prediction"], old=True)
            return JsonResponse({
                "status": True,
                "message": "Successfully tested the system",
                "data": {
                    "old_performance": old_performance,
                    "test_results": test_results
                }
            })

def test_system(condition, stage, test_dataset):
    """
        test the system of the given stage for a participant
        @param participant_id: the condition we are testing
        @param stage: the stage of the system we want to test
        @param test_dataset: the dataset we want to test the system on
    """

    X_test = [item["text"] for item in test_dataset]
    y_test = [item["label"] for item in test_dataset]
    
    system_name = condition.system_name
    participant_id = condition.participant.participant_id
    logger.info(f"participant {participant_id} tests system: {system_name} of the {stage} stage")
    
    latest_system = condition.get_latest_system(stage=stage)
    

    """
        here the training_dataset is used for the LLM ML mixed filter and Trees ML mixed filter
        For other three filter we primarily test, each will read user-related input from the database using both the participant_id and the stage information
    """
    training_dataset = BuildDataSet if stage == "build" else UpdateDataSet 
    status, classifier = latest_system.train(dataset=training_dataset)
    if not status:
        return 0, classifier # the error message
    
    logger.info(f"starting testing for participant {participant_id} at stage {stage} using system {system_name}")
    if system_name == SYSTEMS.PROMPTS_LLM.value:
        from sharedsteps.tasks import test_llm_classifier
        task = test_llm_classifier.delay(classifier.prompts, X_test, y_test)
        return 1, task.id
    else:
        test_results = classifier.test_model(X=X_test, y=y_test)
        return 2, test_results
   
def train_LLM(request):
    logger.info(f"settings.LLM_DEBUG: {settings.LLM_DEBUG}")
    from sharedsteps.tasks import train_llm_task
    logger.info(f"starting training the LLM model")
    request_data = json.loads(request.body)
    
    prompts = request_data.get('instructions')
    dataset = request_data.get('dataset')
        
    task = train_llm_task.delay(prompts, dataset)
    logger.info(f"task {task.id} is started to train the LLM model")
    return JsonResponse({
                    "status": True,
                    "message": f"Successfully started the LLM model training",
                    "data": {
                        "task_id": task.id
                    }
                }, safe=False
            )

def get_LLM_results(request, task_id):
    from celery.result import AsyncResult
    
    task_result = AsyncResult(task_id)
    if task_result.ready():
        logger.info(f"task result {task_result.get()}")
        return JsonResponse({
            "status": True,
            "message": f"Task {task_id} is completed",
            "data": {
                "results": task_result.get()
            }
        })
    else:
        return JsonResponse({
            "status": False,
            "message": "Task is still processing"
        })
    
def get_validate_results(request):
    
    from celery.result import AsyncResult
    # get the parameter of the GET request
    participant_id = request.GET.get('participant_id')
    stage = request.GET.get('stage')
    task_id = request.GET.get('task_id')
    old = request.GET.get('old', default=False)
    
    task_result = AsyncResult(task_id)
    if task_result.ready():
        results = task_result.get()
        
        participant = Participant.objects.get(participant_id=participant_id)
        condition = participant.get_condition(SYSTEMS.PROMPTS_LLM.value)
        if old:
            logger.info(f"save the old test results for participant {participant_id} at stage {stage}")
            condition.save_test_results(stage, results["prediction"], old=True)
        else:
            logger.info(f"save the test results for participant {participant_id} at stage {stage}")
            condition.save_test_results(stage, results["prediction"])
        return JsonResponse({
            "status": True,
            "message": f"Task {task_id} is completed",
            "data": {
                "test_results": results,
            }
        })
    else:
        return JsonResponse({
            "status": False,
            "message": "Task is still processing"
        })

def train_trees(request):
    from systems.trees_filter import TreesFilter

    request_data = json.loads(request.body)
    rules = request_data.get('instructions')
    dataset = request_data.get('dataset')

    dataset = [item["text"] for item in dataset]
    tree_filter = TreesFilter(rules)
    results = tree_filter.test_model(X=dataset, y=None)
    return JsonResponse({
                    "status": True,
                    "message": f"Successfully tested the Trees model",
                    "data": {
                        "results": results
                    }
                }, safe=False
            )

def survey(request):
    # parse out the participant id fr dom the request GET parameters
    participant_id = request.GET.get('participant_id', default=None)
    system_name = request.GET.get('system', default=None)
    stage = request.GET.get('stage', default=None)
    debug = request.GET.get('debug', default="false") == "true"

    status, error_message = utils.check_parameters(participant_id, stage, system_name)
    if not status:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    survey = SurveyDict.get(system_name, "")
    return render(request, 'survey.html', {
        "participant_id": participant_id,
        "system": system_name,
        "stage": stage,
        "survey": survey,
        "debug": debug
    })
    
    