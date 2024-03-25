DEBUG = false;
const SURVEY = true;
let SYSTEM;
let PARTICIPANT_ID;
let STAGE;
let TUTORIAL = false;
const SESSION_TIME = 15;
const MUST_SPENT_TIME = 12;
const TUTORIAL_TIME = 4;

/* time the labeling process */
var interval_id = null;
var is_paused = false;
var time_left = null;
var time_spent = 0;

function complete(timeout=false){
    throw new Error("complete function is not implemented");
}

function save_system(time_spent){
    console.warn("save_system function is not implemented");
}

function countDownTimer(){
    if (!is_paused) {
        time_left = time_left - 1000;
        time_spent = time_spent + 1000;
        // Time calculations for minutes and seconds
        var minutes = Math.floor((time_left % (1000 * 60 * 60)) / (1000 * 60));
        var seconds = Math.floor((time_left % (1000 * 60)) / 1000);

        $("#timer").text(minutes + ":" + (seconds < 10 ? "0" : "") + seconds);

        if(time_spent % 30000 === 0){
            if(!TUTORIAL) save_system(time_spent);
        }
        if(time_spent > MUST_SPENT_TIME * 60 * 1000){
            $('#createFilterButton').removeClass("hidden invisible").removeClass('opacity-50 cursor-not-allowed').addClass('hover:bg-green-700');
        }

        // If the countdown is over, stop the timer
        if (time_left < 0) {
            clearInterval(interval_id);
            $("#timer").text("00:00");
            if(!DEBUG && !TUTORIAL) complete(true);
        }
    }
}

function startTimer(minutes){
    $("#timer").text(`${minutes}:00`);
    if(DEBUG) {
        time_left = minutes * 60 * 1000;
    } else {
        time_left = minutes * 60 * 1000 - time_spent;
    }
    interval_id = setInterval(countDownTimer, 1000);
}

function showLoadingModal(message){
    is_paused = true;
    $('#loadingModal').removeClass('hidden');
    $('#loadingModalText').html(message);
}

function hideLoadingModal(){
    // when called, wait for 2 seconds before hiding the modal
    setTimeout(function(){
        is_paused = false;
        $('#loadingModal').addClass('hidden');  
    }, 1000);

}

let hide_callback = null;
function showAlertModal(message, button="Okay", callback=null){
    console.log("showAlertModal");
    $("#alertModalButton").text(button);
    $("#alertModalText").html(message);
    $("#alertModal").removeClass("hidden");    
    hide_callback = callback;
}

function hideAlertModal(){
    hide_callback = null;
    $("#alertModal").addClass("hidden");
}

function nextAlertModal(){
    $("#alertModal").addClass("hidden");
    if(hide_callback != null){
        hide_callback();
        hide_callback = null;
    }
}

var logs = [];
function addLog(codename, description){
    logs.push({
        timestamp: new Date().toISOString(),
        time_left: time_left,
        codename: codename,
        description: description,
        system: SYSTEM,
    })
}

function logEvents(event, params){
    if(TUTORIAL) return;

    let description = "";
    if(SYSTEM === "examplesML" || SYSTEM === "GroundTruth"){
        switch(event){
            case "Start":
                description = "";
                break;
            case "Complete":
                description = `Finished labeling examples with ${params.positive_number} positive examples out of ${params.label_number} labeled examples.`;
                break;
            case "Filter":
                description = `Switch the "Show only removed examples" to ${params.checked}.`;
                break;
            case "LoadExamples":
                description = `Loaded ${params.dataset_len} more examples.`;
                break;
            case "LabelExamples":
                description = `Labeled this text as ${params.new_label}: "${params.text}"`;
                break;
        }
    }

    // shared events for rulesTrees and promptsLLM
    if(SYSTEM === "rulesTrees" || SYSTEM === "promptsLLM"){
        switch (event) {
            case "Start":
                description = "";
                break;
            case "Complete":
                description = `Completed the configuration of the ${INSTRUCTION_NAME} with ${params.instruction_len} instructions and ${params.dataset_len} examples.`;
                break;
            case "AddNewInstruction":
                description = `Added a new instruction.`;
                break;
            case "DeleteInstruction":
                description = `Deleted the instruction with the id ${params.instruction_id}.`;
                break;
            case "ActivateInstruction":
                description = `(De)activate an instruction with id ${params.instruction_id} to ${params.active}.`;
                break;
            case "LoadMoreData":
                description = `Loaded ${params.batch_size} more data.`;
                break;
            case "ShowExplanation":
                description = `(De)activate the explanation of the comment with id ${params.datum_id} to ${params.show_tooltip}.`;
                break;
        }
    }

    // events for rulesTrees
    if(SYSTEM === "rulesTrees"){
        switch (event) {
            case "TrainTrees":
                description = `Trained the rules with ${params.instruction_len} instructions and ${params.dataset_len} examples.`;
                break;
            case "AddNewUnit":
                description = `Added a new ${params.type} to the instruction ${params.instruction_id}.`;
                break;
            case "RemoveUnit":
                description = `Removed the ${params.type} from the instruction ${params.instruction_id}.`;
                break;
            case "SuggestSynonyms":
                description = `Suggested synonyms for the words: ${params.words}.`;
                break;
            case "SpellingVariants":
                description = `(De)activate the spelling variants of the instruction with id ${params.instruction_id} to ${params.checked}.`;
                break;
            case "AddTagifyPhrase":
                description = `Added a new phrase to the instruction with id ${params.instruction_id}: ${params.phrase}.`;
                break;
            case "RemoveTagifyPhrase":
                description = `Removed a phrase from the instruction with id ${params.instruction_id}: ${params.phrase}.`;
                break;
        }
    }

    if(SYSTEM === "promptsLLM"){
        switch(event){
            case "TrainLLM":
                description = `Trained the LLM with ${params.instruction_len} instructions and ${params.dataset_len} examples with the task ${params.task_id}.`;
                break;
            case "LLMResults":
                description = `Received the predictions from the backend for the task ${params.task_id}.`;
                break;
            case "RephraseInstruction":
                description = `Rephrased the instruction with id ${params.instruction_id} from ${params.instruction} to ${params.rephrase_instruction}.`;
                break;
            case "AcceptNewPhrases":
                description = `Accepted the new phrases for the instruction with id ${params.instruction_id}.`;
                break;
            case "RejectNewPhrases":
                description = `Rejected the new phrases for the instruction with id ${params.instruction_id}.`;
                break;
            case "CopyText":
                description = `Copied the text: ${params.text}.`;
                break;
        }
    }

    addLog(event, description);
}