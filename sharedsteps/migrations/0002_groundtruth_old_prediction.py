# Generated by Django 4.2.6 on 2024-02-13 06:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sharedsteps', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='groundtruth',
            name='old_prediction',
            field=models.IntegerField(null=True),
        ),
    ]