# Generated by Django 4.2.6 on 2024-01-17 22:47

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ExampleLabel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('participant_id', models.CharField(max_length=100)),
                ('stage', models.CharField(choices=[('build', 'build'), ('update', 'update')], max_length=10)),
                ('text', models.TextField()),
                ('label', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='GroundTruth',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('participant_id', models.CharField(max_length=100)),
                ('text', models.TextField()),
                ('label', models.IntegerField()),
                ('type', models.CharField(choices=[('validation', 'validation'), ('test', 'test')], max_length=20)),
            ],
        ),
        migrations.CreateModel(
            name='Participant',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('participant_id', models.CharField(max_length=100)),
                ('random_prime', models.IntegerField(null=True)),
                ('current_batch', models.IntegerField(default=0)),
                ('system', models.CharField(choices=[('PROMPTS_LLM', 'promptsLLM'), ('PROMPTS_ML', 'promptsML'), ('EXAMPLES_ML', 'examplesML'), ('RULES_TREES', 'rulesTrees'), ('RULES_ML', 'rulesML')], max_length=100, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='PromptWrite',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('participant_id', models.CharField(max_length=100)),
                ('stage', models.CharField(choices=[('build', 'build'), ('update', 'update')], max_length=10)),
                ('prompt_id', models.IntegerField()),
                ('rubric', models.TextField()),
                ('priority', models.IntegerField()),
                ('positives', models.TextField()),
                ('negatives', models.TextField()),
                ('action', models.IntegerField(choices=[(0, 'approve'), (1, 'remove')])),
            ],
        ),
        migrations.CreateModel(
            name='RuleConfigure',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('participant_id', models.CharField(max_length=100)),
                ('stage', models.CharField(choices=[('build', 'build'), ('update', 'update')], max_length=10)),
                ('name', models.TextField()),
                ('rule_id', models.IntegerField()),
                ('priority', models.IntegerField()),
                ('variants', models.BooleanField()),
                ('action', models.IntegerField(choices=[(0, 'approve'), (1, 'remove')])),
            ],
        ),
        migrations.CreateModel(
            name='RuleUnit',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('type', models.CharField(choices=[('include', 'include'), ('exclude', 'exclude')], max_length=10)),
                ('words', models.TextField()),
                ('rule', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='units', to='sharedsteps.ruleconfigure')),
            ],
        ),
    ]
