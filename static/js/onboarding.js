'use strict';

(function () {
  var conditions = [
    // 'basic-filter.taskA',
    // 'basic-filter.taskB',
    'basic-ML.taskA',
    'basic-ML.taskB',
    // 'advanced-ML.taskA',
    // 'advanced-ML.taskB',
    // 'forest-filter.taskA',
    // 'forest-filter.taskB',
  ]

  function $(e) {
    return document.getElementById(e);
  }

  function generateParticipantId() {
    var randComponent = Math.floor(Math.random() * 10000);
    var cond = Math.floor(Math.random() * conditions.length);
    return btoa(randComponent + '|' + cond + '|' + randComponent + '|' + Math.round(Date.now()) % 10000);
  }

  function getParticipantCond(id) {
    try {
      var values = atob(id.trim()).split('|');
    } catch (e) {
      return null;
    }

    if (values.length !== 4) {
      return null;
    }
    if (values[0] !== values[2]) {
      return null;
    }
    return conditions[parseInt(values[1], 10)];
  }

  function triggerChange() {
    try {
      var form = document.getElementById('main-form');
      $('more-new').style.display = 'none';
      $('more-existing').style.display = 'none';
      $('more-' + form.mode.value).style.display = '';
    } catch (e) {}
  }

  window.addEventListener('load', function () {
    var form = document.getElementById('main-form');

    triggerChange();

    $('participant-id').innerText = generateParticipantId();

    $('radio-choice-new').addEventListener('click', function (e) {
      triggerChange();
    });
    $('radio-choice-existing').addEventListener('click', function (e) {
      triggerChange();
    });

    $('start-button').addEventListener('click', function (e) {
      e.preventDefault();

      if (!(form.mode.value === 'new') && !(form.mode.value === 'existing')) {
        alert('You need to select an option!');
        return;
      } else if (form.mode.value === 'new') {
        form.participantId.value = $('participant-id').innerText.trim();
      } else if (form.mode.value === 'existing') {
        form.participantId.value = $('participant-id-input').value.trim();
      }

      
      var participantId = form.participantId.value;
      
      var condition = getParticipantCond(participantId);
      if (condition === null) {
        alert('Something wrong with your participant id. Please check that you have the right value.');
        return;
      } 

      form.condition.value = condition;
      form.participantId.value = participantId;

      form.submit();
    });

  });
})();