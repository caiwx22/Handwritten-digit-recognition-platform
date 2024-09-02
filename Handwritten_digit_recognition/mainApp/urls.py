from django.urls import path
from mainApp import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", views.defaultJump),
    path("index/", views.hello),
    path("training/", views.training),
    path("results/", views.results),
    path("getParameter/", views.getParameter),
    path("deleteTask/", views.deleteInfo),
    path("useModel/", views.useModel),
    path('upload/', views.uploadImage),
    path('save_signature/', views.save_signature),
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
