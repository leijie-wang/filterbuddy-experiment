'use strict'

// we assume every single quote is properly escaped; this should be ensured when adding them to the list
// for now, we only require users to input one example for each category
function escape_single_quote(text, remove_newline=false) {
    text = text.trim();
    if (remove_newline) {
        text = text.replace(/\r?\n/g, "");
    }
    return text.replace(/'/g, "\\'");
}

function escape_html(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
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

function post_backend(url, data, success_function=null){
    console.log("start POST to " + url);
    const csrftoken = document.querySelector('[name=csrfmiddlewaretoken]').value
    return $.ajax({
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

function redirect(url, parameters, new_tab=false){
    let new_url = url + '?';
    // iterature through the dict and add parameters to the url
    Object.keys(parameters).forEach(function(key) {
        new_url += `${key}=${parameters[key]}&`;
    });
    new_url = new_url.slice(0, -1); // remove the last '&' or "?" when there is no parameter
    if (new_tab) {
        window.open(new_url, '_blank');
    } else {
        window.location.href = new_url;
    }
}

function diff_wordMode(text1, text2) {
    var dmp = new diff_match_patch();
    var a = dmp.diff_wordsToChars_(text1, text2);
    var lineText1 = a.chars1;
    var lineText2 = a.chars2;
    var lineArray = a.lineArray;
    var diffs = dmp.diff_main(lineText1, lineText2, false);
    dmp.diff_charsToLines_(diffs, lineArray);
    dmp.diff_cleanupSemantic(diffs);
    return diffs;
}


