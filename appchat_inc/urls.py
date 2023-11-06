from django.urls import path
from . import views
from . import apis

urlpatterns = [
    path("", views.index, name="index"),
    path("temp_index", views.template_index, name="template_index"),
    path("chat_thebot",views.chat_thebot ,name="chat to the bot"),
    path("permi_calculation",views.permi_calculation ,name="premi calculation"),
    path('chatlog/',views.chatlog, name='chatlog'),
    path('getdesc_chatlog/',views.getdesc_chatlog, name='getdesc_chatlog'),
    
    path('knowledge/',views.knowledge, name='knowledge'),
    path('knowledge_setup/',views.knowledge_setup, name='knowledge setup'),

    path("chat_thebot_api",apis.chat_thebot_api ,name="api chat to the bot"),
    path("knowledge_setup_api",apis.knowledge_setup_api ,name="knowledge_setup_api"),
    path("get_chatlog_api",apis.get_chatlog_api ,name="get_chatlog_api"),

    path('signin/',views.signin, name='signin'),
    path('signout/',views.signout, name='signout'),
    path('signup/',views.signup, name='signup'),
    # path('profile/',views.profile, name='profile'),
    path('getchatlog', views.OrderListJson.as_view(), name='getchatlog'),
    path('get_chatlog', views.get_chatlog, name='get_chatlog'),
    path('get_dropdown_category', views.get_dropdown_category, name='get_dropdown_category'),
    path('get_dropdown_model', views.get_dropdown_model, name='get_dropdown_model'),

    # path('getchatlog/', views.getchatlog, name='getchatlog'),
]