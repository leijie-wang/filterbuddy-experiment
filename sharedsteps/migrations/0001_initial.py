# Generated by Django 4.2.6 on 2023-12-20 19:18

from django.db import migrations, models


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
                ('text', models.TextField()),
                ('label', models.IntegerField()),
            ],
        ),
    ]