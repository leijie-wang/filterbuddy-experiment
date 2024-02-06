from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'experimentweb.settings')

app = Celery('experimentweb')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# Don't store task results in the database
app.conf.task_ignore_result = True
