# Generated by Django 4.2.6 on 2024-02-26 17:12

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
            name='ExperimentLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('participant_id', models.CharField(max_length=100)),
                ('timestamp', models.DateTimeField()),
                ('time_left', models.IntegerField()),
                ('stage', models.CharField(choices=[('build', 'build'), ('update', 'update')], max_length=10)),
                ('system', models.CharField(max_length=100)),
                ('codename', models.CharField(max_length=100)),
                ('description', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='GroundTruth',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('participant_id', models.CharField(max_length=100)),
                ('text', models.TextField()),
                ('label', models.IntegerField()),
                ('stage', models.CharField(choices=[('build', 'build'), ('update', 'update')], max_length=10)),
                ('prediction', models.IntegerField(null=True)),
                ('old_prediction', models.IntegerField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Participant',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('participant_id', models.CharField(max_length=100)),
                ('random_seed', models.IntegerField(null=True)),
                ('stage', models.CharField(choices=[('build', 'build'), ('update', 'update')], max_length=10)),
                ('system', models.CharField(choices=[('PROMPTS_LLM', 'promptsLLM'), ('PROMPTS_ML', 'promptsML'), ('EXAMPLES_ML', 'examplesML'), ('RULES_TREES', 'rulesTrees'), ('RULES_ML', 'rulesML')], max_length=100, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='PromptWrite',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('participant_id', models.CharField(max_length=100)),
                ('stage', models.CharField(choices=[('build', 'build'), ('update', 'update')], max_length=10)),
                ('name', models.TextField()),
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
