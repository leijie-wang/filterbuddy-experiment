'use strict'

// we assume every single quote is properly escaped; this should be ensured when adding them to the list
// for now, we only require users to input one example for each category
function escape_single_quote(text) {
    return text.replace(/'/g, "\\'");
}

function capitalize(str) {
    if (str && typeof str === 'string') {
        return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
    }
    return str;
}

function to_percentage(num) {
    const percentage = (num * 100).toFixed(0);
    return percentage.padStart(2, '0') + "%";
}

function remove_unit_section(event, class_name) {
    $(event.target).closest(`.${class_name}`).remove();
}

function removeNearestAncestor(element, class_name){
    let parent = element.closest(`.${class_name}`);
    if (parent == null){
        console.log(`cannot find ancestor with class name ${class_name}`);
    } return parent.remove();
}   

function get_backend(url, data, success_function){
    console.log("start GET to " + url);
    $.ajax({
        url: url,
        type: "GET",
        data: data,
        dataType: 'json',
        success: success_function,
    });
}

function post_backend(url, data, success_function){
    console.log("start POST to " + url);
    const csrftoken = document.querySelector('[name=csrfmiddlewaretoken]').value
    $.ajax({
        url: url,
        type: "POST",
        data: JSON.stringify(data),
        contentType: 'application/json',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrftoken
        },
        success: success_function,
    });
}

function redirect(url, parameters){
    let new_url = url + '?';
    // iterature through the dict and add parameters to the url
    Object.keys(parameters).forEach(function(key) {
        new_url += `${key}=${parameters[key]}&`;
    });
    new_url = new_url.slice(0, -1); // remove the last '&' or "?" when there is no parameter
    window.location.href = new_url;
}

function display_data(new_dataset, dataset, new_separator=true){
    // display data on the page, this is not for users to label data
    let textList = $(`#textList`);

    if (new_separator) {
        /* remove the old separator for new data; and add a new separator before adding new data */
        textList.find('.newSeparator').remove();

        const separator_html = `
            <div class="relative mb-3 newSeparator">
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
        let text_div_html = `
            <div class="flex flex-col gap-y-1 mb-2" x-data="{ showTooltip: false }">   
                <div id="datum-${i + start_index}" 
                    class="datum flex flex-row py-1 px-2 self-stretch unLabel"
                >
                    <div class="flex-grow w-7/10 px-2 py-1 self-center">${new_dataset[i].text}</div>
                    <div class="self-end" @click="showTooltip = !showTooltip">
                        <i class="fa-solid fa-circle-info fa-xs" style="color: #868788;"></i>
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

    /*
        even though the dataset parameter as a dict is passed by reference, we still need to return it to make sure the caller gets the updated dataset
        otherwise, mysteriously, the dataset as a global variable is not updated
    */
    return dataset;
}

function display_labeling_data(new_dataset, dataset, new_separator=true){
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
            <div id="datum-${index}" class="flex flex-row items-center space-x-1 py-1 border-b border-gray-300">   
                <div id="text-${index}" class="flex-grow w-7/10 p-2">${new_dataset[i].text}</div>
                <div class="flex flex-row w-3/10 justify-center space-x-2">
                    <button 
                        @click="change_label(${index}, true);" 
                        class="text-white font-bold py-1 px-3 rounded unselected-button yes-button"
                    >
                        Toxic
                    </button>
                    <button
                        @click="change_label(${index}, false);" 
                        class="text-white font-bold py-1 px-3 rounded unselected-button no-button"
                    >
                        No
                    </button>
                </div>
            </div>`
        textList.append(text_div_html);
    }
    // :class="{'bg-blue-500 hover:bg-blue-600': selected == true, 'bg-gray-300 hover:bg-gray-400': selected == false}" 
    dataset = dataset.concat(new_dataset);
    return dataset;
}