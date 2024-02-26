const DEBUG = true;
let SYSTEM;
let PARTICIPANT_ID;
let STAGE;

/* time the labeling process */
var interval_id = null;
var is_paused = false;
var time_left = null;

function complete(timeout=false){
    throw new Error("complete function is not implemented");
}

function countDownTimer(){
    if (!is_paused) {
        time_left = time_left - 1000;
        // Time calculations for minutes and seconds
        var minutes = Math.floor((time_left % (1000 * 60 * 60)) / (1000 * 60));
        var seconds = Math.floor((time_left % (1000 * 60)) / 1000);

        $("#timer").text(minutes + ":" + (seconds < 10 ? "0" : "") + seconds);

        // If the countdown is over, stop the timer
        if (time_left < 0) {
            clearInterval(interval_id);
            $("#timer").text("00:00");
            complete(true);
        }
    }
}

function startTimer(minutes){
    $("#timer").text(`${minutes}:00`);
    time_left = minutes * 60 * 1000;
    interval_id = setInterval(countDownTimer, 1000);
}

var logs = [];
function addLog(message){
    logs.push({
        timestamp: new Date().toISOString(),
        time_left: time_left,
        message: message
    })
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
    $("#alertModalButton").text(button);
    $("#alertModalText").html(message);
    $("#alertModal").removeClass("hidden");    
    hide_callback = callback;
}

function hideAlertModal(){
    if (hide_callback != null){
        hide_callback();
        hide_callback = null;
    }
    $("#alertModal").addClass("hidden");
}


