"""
URL configuration for experimentweb project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from sharedsteps import views as sharedviews

urlpatterns = [
    path('admin/', admin.site.urls),
    path('onboarding/', sharedviews.onboarding),

    path('groundtruth/', sharedviews.label_ground_truth),
    path('store_groundtruth/', sharedviews.store_groundtruth),
    path('test_filter/', sharedviews.test_filter),

    path('load_system/', sharedviews.load_system),
    
    # path('load_more_data/', sharedviews.load_more_data),
    path('validate_page/', sharedviews.validate_page),
    
    path('examplelabel/', sharedviews.examplelabel),
    path('store_labels/', sharedviews.store_labels),

    path('promptwrite/', sharedviews.promptwrite),
    path("store_prompts/", sharedviews.store_prompts),
    path('trainLLM/', sharedviews.train_LLM),
    path('get_LLM_results/<str:task_id>/', sharedviews.get_LLM_results),
    path('get_validate_results/', sharedviews.get_validate_results),

    path('ruleconfigure/', sharedviews.ruleconfigure),
    path('store_rules/', sharedviews.store_rules),
    path('trainTrees/', sharedviews.train_trees),
]
