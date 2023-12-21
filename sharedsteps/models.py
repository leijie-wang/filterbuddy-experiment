from django.db import models

# Create your models here.
class ExampleLabel(models.Model):
    participant_id = models.CharField(max_length=100)
    text = models.TextField()
    label = models.IntegerField()

    def __str__(self):
        return f"Participant {self.participant_id} labeled {self.text} as {self.label}"