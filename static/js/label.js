var dataset = [];
var positive_number = 0; // how many examples are labeled as toxic   
var label_number = 0; // how many examples are labeled

function displayLabelingData(new_dataset, new_separator=true){
    // display data on the page, here we need to add buttons for users to label data
    let textList = $('#textList');

    if (new_separator) {
        /* remove the old separator for new data; and add a new separator before adding new data */
        textList.find('.newSeparator').remove();

        const separator_html = `
            <div class="relative my-4 newSeparator bg-blue-200">
                <div class="w-full border-t border-red-400"></div>
                <div class="text-red-400 absolute -mt-4 bg-white px-1 font-medium text-2xl" style="left: 50%; transform: translateX(-50%);">
                    New
                </div>
            </div>`;

        textList.append(separator_html);
    }
    
    /* start adding new data */
    console.log(`start displaying ${new_dataset.length} new data`);
    let start_index = dataset.length;
    for (let i = 0; i < new_dataset.length; i++) {
        // create a new div element for each text with the template below
        let index = i + start_index;
        let text_div_html = `
            <div id="datum-${index}" x-data="{label: null}" class="flex flex-row items-center space-x-1 py-1 border-b border-gray-300">   
                <div id="text-${index}" class="flex-grow max-w-[80%] p-2">${escape_html(new_dataset[i].text)}</div>
                <div class="grow-0 flex flex-row w-fit justify-center space-x-2">
                    <button 
                        @click="label = true; changeLabel(${index}, true);"
                        :disabled="label == true"
                        class="text-white py-1 px-3 rounded unselected-button yes-button"
                    >
                        Remove
                    </button>
                    <button
                        @click="label = false; changeLabel(${index}, false);" 
                        :disabled="label == false"
                        class="text-white py-1 px-3 rounded unselected-button no-button"
                    >
                        Keep
                    </button>
                </div>
            </div>`
        textList.append(text_div_html);
    }
    // :class="{'bg-blue-500 hover:bg-blue-600': selected == true, 'bg-gray-300 hover:bg-gray-400': selected == false}" 
    dataset = dataset.concat(new_dataset);
    if(new_separator){
        textList.find('.newSeparator').last().get(0).scrollIntoView({behavior: "smooth"});
    }
}

function changeLabel(index, new_label){
    if(new_label === dataset[index].label) return;


    logEvents("LabelExamples", {new_label: new_label, text: dataset[index].text, new: dataset[index].label == null});

    positive_number = positive_number + (new_label === true ? 1 : (dataset[index].label === true ? -1 : 0));
    label_number = label_number + (dataset[index].label == null ? 1 : 0);
    $('#positive-number').text(positive_number);
    $('#label-number').text(label_number);


    dataset[index].label = new_label;
    if(new_label){
        $(`#datum-${index} .yes-button`).removeClass('unselected-button').addClass('selected-button');
        $(`#datum-${index} .no-button`).removeClass('selected-button').addClass('unselected-button');
    } else {
        $(`#datum-${index} .no-button`).removeClass('unselected-button').addClass('selected-button');
        $(`#datum-${index} .yes-button`).removeClass('selected-button').addClass('unselected-button');

    }
}