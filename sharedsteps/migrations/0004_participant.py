# Generated by Django 4.2.6 on 2024-01-06 04:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sharedsteps', '0003_promptwrite_prompt_id'),
    ]

    operations = [
        migrations.CreateModel(
            name='Participant',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('participant_id', models.CharField(max_length=100)),
                ('random_prime', models.IntegerField()),
                ('system', models.CharField(choices=[('PROMPTS_LLM', 'promptsLLM'), ('PROMPTS_ML', 'promptsML'), ('EXAMPLES_ML', 'examplesML')], max_length=100)),
            ],
        ),
    ]
