import json
from django.shortcuts import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from . import views

ACCEPTED_KEY = "castbuddy-thejanuslab"

def check_api_key(request):
    "' check api key '"
    if request.POST.get('api_key'):
        api_key = request.POST.get('api_key')
        if api_key == ACCEPTED_KEY:
            return True

    message = {}
    message['status'] = False
    message['message'] = 'Api key not recognized'
    return HttpResponse(json.dumps(message), content_type="application/json")

@csrf_exempt
def chat_thebot_api(request):
    "' chat the bot '"

    check_key = check_api_key(request)

    if check_key is True:
        return views.chat_thebot(request)
    else:
        return check_key

@csrf_exempt
def knowledge_setup_api(request):
    "' upload knowledge '"

    check_key = check_api_key(request)

    if check_key is True:
        return views.knowledge_setup(request)
    else:
        return check_key   

@csrf_exempt
def get_chatlog_api(request):
    "' get chatlog '"

    check_key = check_api_key(request)

    if check_key is True:
        return views.get_chatlog(request)
    else:
        return check_key