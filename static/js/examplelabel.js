


function display_data(new_dataset){
    // display data on the page
    let textList = $('#textList');

    for (let i = 0; i < new_dataset.length; i++) {
        // create a new div element for each text with the template 

        console.log(new_dataset[i]);
        let newDiv = document.createElement('div');
        newDiv.innerHTML = `
            <div 
                x-data="{selected: false, text: '${new_dataset[i]}'}" 
                x-show="!filtered || selected == true" 
                class="flex flex-row items-center space-x-4 py-1 border-b border-gray-300"
            >
                <div x-text="text" class="flex-grow w-7/10 p-2"></div>
                <div class="flex flex-row w-3/10 justify-center space-x-2">
                    <button 
                        @click="selected = !selected; yes_counter += selected ? 1 : -1" 
                        :class="{'bg-blue-500 hover:bg-blue-600': selected == true, 'bg-gray-300 hover:bg-gray-400': selected == false}" 
                        class="text-white font-bold py-1 px-3 rounded"
                    >
                        Yes
                    </button>
                </div>
            </div>`;
        textList.append(newDiv);
    }
}

display_data(dataset);

function trainML(){
    // fetch information from examplelabel.html and send to server
    let data = [];
    $('#textList > div').each(function () {
        // read its x-data from alphine framework
        let x = $(this).attr('x-data');
        // the x-data is in the format of "{selected: null, text: '....'}" in html, extract its text and selected status

        
    });

}

