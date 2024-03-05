const BATCH_SIZE = 30;
var COMPLETE_DATASET = [];
var INSTRUCTION_NAME = "instruction";
var dataset = [];

var current_batch = 0; 

var instructions = {};
var instruction_counter = 0; // indicate the highest id that has been used

var selected_instructions = [];
var selected_prediction = "all";

let new_instruction_template;

function updateName(id, text){
    instructions[id].name = text;
}

function addNewInstruction(){
    logEvents("AddNewInstruction", {});
    // read from configuring instruction panel and add this new instruction to the instructions in jquery
    if(INSTRUCTION_NAME === "rule"){
        displayInstructions([{
            "action": 1,
            "name": "",
            "units": [{
                "type": "include",
                "words": [],
            }]   
        }]);
    } else if (INSTRUCTION_NAME === "prompt"){
        displayInstructions([{
            "name": "",
            "action": 1,
            "rubric": "",
            "positives": [],
            "negatives": []
        }]);
    }
}

function deleteInstruction(deleted_id) {
    logEvents("ActivateInstruction", {instruction_id: deleted_id});

    delete instructions[deleted_id];
    $(`#instruction-${deleted_id}`).remove();

    // remove the instruction from selected_prompts if needed
    selected_instructions = selected_instructions.filter((id) => id != deleted_id);
}

function copyText(index){
    navigator.clipboard.writeText(dataset[index].text);
    logEvents("CopyText", {text: dataset[index].text});
}

function displayData(new_dataset, new_separator=true){
    let textList = $(`#textList`);

    if (new_separator) {
        /* remove the old separator for new data; and add a new separator before adding new data */
        textList.find('.newSeparator').remove();

        const separator_html = `
            <div class="relative py-3 newSeparator">
                <div class="w-full border-t border-gray-300"></div>
                <div class="text-red-400 absolute -mt-4 bg-white px-1 font-medium text-lg" style="left: 50%; transform: translateX(-50%);">
                    New
                </div>
            </div>`;

        textList.append(separator_html);
    }

    /* start adding new data */
    console.log(`start displaying ${new_dataset.length} new data`);
    let start_index = dataset.length; // used as the id of the datum
    for (let i = 0; i < new_dataset.length; i++) {
        /* Here i use mb-2 instead of adding a gap-y-2 to the textList div because the latter will still keep gaps for hidden elements */
        let copy_button = "";
        if(INSTRUCTION_NAME === "prompt"){
            copy_button = `
                <div x-data="{ copied: false, isHovering: false}" 
                    @click="copied = true; copyText(${i+start_index}); setTimeout(() => copied = false, 500);" 
                    @mouseenter="isHovering = true" 
                    @mouseleave="isHovering = false" 
                    class="tooltipEnabled cursor-pointer"
                    title="Copy the example"
                >
                    <i class="fa-solid fa-copy text-stone-400 fa"></i>
                    <div x-show="copied" 
                        class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 py-2 px-4 text-white bg-gray-600 text-sm rounded">
                        Copied to Your Clipboard
                    </div>
                </div>
            `;
        }
        let text_div_html = `
            <div class="flex flex-col gap-y-1 mb-2" x-data="{ showTooltip: false }">   
                <div id="datum-${i + start_index}" 
                    class="datum flex flex-row py-1 px-2 self-stretch unLabel relative"
                >
                    <div class="datumText flex-grow w-7/10 px-2 py-1 self-center">${escape_html(new_dataset[i].text)}</div>
                    <div class="self-end flex gap-x-2">
                        ${copy_button}
                        <div 
                            class="tooltipEnabled" 
                            title="See explanation"
                            @click="showTooltip = !showTooltip; logEvents('ShowExplanation', {datum_id: ${i + start_index}, show_tooltip: showTooltip});"
                        >
                            <i class="fa-solid fa-circle-info text-gray-500 fa"></i>
                        </div>
                    </div>
                </div>
                <div 
                    class="flex flex-col px-2 pb-1 self-end max-w-[75%] rounded unLabel"
                    x-show="showTooltip"
                    id="datumTooltip-${i + start_index}"
                >
                    <div class="tooltip-arrow"></div>
                    <div class="text-sm">
                        This comment is unlabeled yet.
                    </div>
                </div>
            </div>`;
        textList.append(text_div_html);
    }
    dataset = dataset.concat(new_dataset);
    $(`#countDisplay`).text(`There are now ${dataset.length} comments in total`);

    if(new_separator){
        textList.find('.newSeparator').last().get(0).scrollIntoView({behavior: "smooth", block: "start", inline: "nearest"});
    }
}

function loadMoreData(new_separator=true){
    
    let start = current_batch * BATCH_SIZE;
    let end = (current_batch + 1) * BATCH_SIZE;
    if(end >= COMPLETE_DATASET.length) end = COMPLETE_DATASET.length;
    let new_data = COMPLETE_DATASET.slice(start, end);
    logEvents("LoadMoreData", {batch_size: new_data.length});

    displayData(new_data, new_separator);
    current_batch++;
}

function showFilteredData(){
    /* iterature through dataset and hide the datum that doesn't match the filter selected_prediction and selected_prompts */

    let expected_prediction = selected_prediction === "removed" ? [1] : selected_prediction === "approved" ? [0, null] : [1, 0, null, undefined];
    let counter = 0;
    let targeted_counter = 0;
    let approved_counter = 0;
    let removed_counter = 0;

    // console.log("selected instructions", selected_instructions);
    // console.log("selected_predictions", selected_prediction)
    dataset.forEach((datum, index) => {
        let show = expected_prediction.includes(datum.total_prediction);
        
        if (show) 
            $(`#datum-${index}`).parent().show();
        else
            $(`#datum-${index}`).parent().hide();

        counter++;
        targeted_counter += datum.total_prediction !== null ? 1 : 0;
        approved_counter += datum.total_prediction === 0 ? 1 : 0;
        removed_counter += datum.total_prediction === 1 ? 1 : 0;
    });

    let text = `Your filter will remove ${removed_counter} out of ${dataset.length} comments in total.`;
    $('#countDisplay').text(text);

}

function generateExplanation(datum, datum_index){
    /* generate the explanation for each prediction made by the LLMs */
    if (datum.total_prediction === null || datum.total_prediction === 0){
        if(INSTRUCTION_NAME == "rule") {
            return `This comment will not be removed by any of your ${INSTRUCTION_NAME}s.`;
        }
    }
    else {
        let explanation = `<p>This comment will be removed by the following ${INSTRUCTION_NAME}s :</p><ul class='list-none pl-0 text-gray-600'>`;

        let highest_priority_id = null;
        let highest_priority = 0;
        for (let i = 0; i < datum.predictions.length; i++) {
            const pred = datum.predictions[i];
            if (pred.prediction !== null){
                let instruction = instructions[pred.id];
                
                explanation += `
                    <li class="before:content-['•'] before:mr-2">
                        ${capitalize(INSTRUCTION_NAME)} <span class="font-medium text-emerald-600 italic font-serif">${instruction.name}</span>
                    </li>`;
                if (highest_priority_id === null){
                    highest_priority = i;
                    highest_priority_id = pred.id;
                }
            }
        }
        explanation += "</ul>";
        return explanation;
    }
}

function updateDataLabels(results) {
    /* update the labels of the data according to the predictions given by LLMs */
    let prediction = results["prediction"];
    let texts_predictions = results["texts_predictions"];

    for (let i = 0; i < prediction.length; i++) {
        dataset[i].total_prediction = prediction[i];
        dataset[i].predictions = texts_predictions[i];
        
        // update the label according to the total prediction
        let new_class = dataset[i].total_prediction === null ? "unLabel" : dataset[i].total_prediction ? "trueLabel" : "falseLabel";
        const datum = $(`#datum-${i}`);
        datum.removeClass("unLabel trueLabel falseLabel");
        datum.addClass(new_class);

        // select the tooltip
        const tooltip_div = $(`#datumTooltip-${i}`);
        tooltip_div.removeClass("unLabel trueLabel falseLabel");
        tooltip_div.addClass(new_class);

        // update the tooltip content
        let explanation = generateExplanation(dataset[i], i);
        // set the second child of the tooltip div to be the explanation, use innerhtml

        tooltip_div.children().eq(1).html(explanation);
    }
}

$(document).ready(function(){
    $("#predictionFilter").on("change", function() {
        selected_prediction = $(this).val();
        addLog(`[${SYSTEM}: PredictionFilter] Changed the prediction filter to ${selected_prediction}.`);
        showFilteredData();
    });
});