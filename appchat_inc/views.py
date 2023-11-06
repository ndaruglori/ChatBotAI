from django.shortcuts import render, HttpResponse, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import logout
import json
import datetime
from django.http import JsonResponse
from django_datatables_view.base_datatable_view import BaseDatatableView
from django.utils.html import escape
from django.core.paginator import Paginator
from django.core import serializers
from appchat_inc.models import Chatlog
from appchat_inc.models import FileModel
from appchat_inc.models import Vehicle_Master
from appchat_inc.models import Plat_No
import re
#from appchat_inc.models import LocationModel
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Qdrant
import qdrant_client
import openai
import requests

from langchain.llms import HuggingFaceHub

import os
import sys

from loguru import logger
from langchain.callbacks import get_openai_callback
# from langchain.callbacks import FileCallbackHandler
import langchain

from langchain import LLMChain, PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate

from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain


from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains import ReduceDocumentsChain
from langchain.chains import MapReduceDocumentsChain
import tiktoken
from langchain.docstore.document import Document


docs_arr = ["Knowledgebase - 500", 
            "Knowledgebase - 750",
            "Knowledgebase - 1000",
            "Hitung Premi - 1000"
            ]

logfile = "log/output-{time:YYYY-MM-DD}.log"
# log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS zz} | {message}"
# logFormat = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <b>{message}</b>"
logger.remove()
logger.add(sys.stdout, format=log_format)
logger.add(logfile, rotation="10 MB", format=log_format)
# handler = FileCallbackHandler(logfile)

# cara 1
# import logging
# logging.basicConfig(filename="log.txt", level=logging.DEBUG,
#                     format="%(asctime)s %(message)s", filemode="w")

# cara 2
# class CustomLogger:
 
#     def __init__(self, filename):
#         self.console = sys.stdout
#         self.file = open(filename, 'w')
 
#     def write(self, message):
#         self.console.write(message)
#         self.file.write(message)
 
#     def flush(self):
#         self.console.flush()
#         self.file.flush()

# time = datetime.datetime.now()
# path = 'log/out-'+str(time)+'.log'

# sys.stdout = CustomLogger(path)


# Create your views here.
def index(request):   
    return render(request,'home.html')

def template_index(request):
    return render(request,'html/index.html')

def chatlog(request):
    return render(request,'chatlog.html')

def int_or_0(value):
    "'make use the value is integer'"
    try:
        return int(value)
    except:
        return 0

def knowledge_setup(request):
    if request.method == 'GET':
        return render(request,'knowledge_setup.html')
    elif request.method == 'POST':
        recreate = request.POST.get('recreate')
        chunk_size = request.POST.get('chunk_size')
        chat_knowledge = request.POST.get('chat_knowledge', '1000')

        print('chat_knowledge '+ chat_knowledge)

        chunk_size = int_or_0(chunk_size)
       
        selected_vec = ''

        if chat_knowledge == "Hitung Premi":
            selected_vec = docs_arr[3]
        else:
            if chunk_size == 500:
                selected_vec = docs_arr[0]
            elif chunk_size == 750:
                selected_vec = docs_arr[1]
            else:
                selected_vec = docs_arr[2]

       
        if recreate == 'true':
            recreate = True
        elif recreate == 'false':
            recreate = False

        print('upload to ' + selected_vec)
        print(recreate)
        print(chunk_size)
        # handle_uploaded_file(request.FILES["filetxt"])
        txt = handle_uploaded_file(request, recreate, str(chunk_size))
        chunks = get_text_chunks_recursive(txt, chunk_size)

        status = False
        msg = ''

        try:
            save_to_qdrant(chunks, selected_vec, recreate)
            status = True
            msg = 'Upload success'
        except TimeoutError:
            status = False
            msg = 'Time out'
        except:
            status = False
            msg = 'Upload error'

        message = {}
        message['status'] = status
        message['message'] = msg
        return HttpResponse(json.dumps(message), content_type="application/json")
    else:
        return render(request, 'knowledge_setup.html')
    
def handle_uploaded_file(request, recreate, model_type):
    """save file and read it"""
    file_model = FileModel()
    _, file = request.FILES.popitem()  # get first element of the uploaded files
    file = file[0]  # get the file from MultiValueDict
    file_model.file = file
    file_model.recreate = recreate
    file_model.model_type = model_type
    file_model.save()

    # file_path = request.build_absolute_uri(file_model.file.url)
    # with open(file_path, encoding='utf8') as f:
    contents = ''
    for line in file:
        contents += str(line.strip(),"utf-8") + '\n'
    return contents


def get_text_chunks_recursive(text, size = 500, overlap = 100):
    """splitting the text to multiple chunk"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function = len
    )
    chunks = text_splitter.split_text(text)

    # for i, _ in enumerate(chunks):
    #     logger.info(f"chunk {i}, size: {len(chunks[i])}")
    #     logger.info(f"{chunks[i]}")
    #     logger.info("--------")
    return chunks

def save_to_qdrant(text_chunks, selected_doc = None, recreate = False):
    """recreate and save to qdrant"""
    load_dotenv()
    embeddings = OpenAIEmbeddings()

    if selected_doc == docs_arr[0]:
        host = os.getenv("QDRANT_HOST")
        api_key = os.getenv("QDRANT_API_KEY")
        collection_name=os.getenv("QDRANT_COLLECTION_NAME")

    elif selected_doc == docs_arr[1]:
        host = os.getenv("QDRANT_HOST-2")
        api_key = os.getenv("QDRANT_API_KEY-2")
        collection_name=os.getenv("QDRANT_COLLECTION_NAME-2")

    elif selected_doc == docs_arr[2]:
        host = os.getenv("QDRANT_HOST-3")
        api_key = os.getenv("QDRANT_API_KEY-3")
        collection_name=os.getenv("QDRANT_COLLECTION_NAME-3")

    elif selected_doc == docs_arr[3]:
        host = os.getenv("QDRANT_HOST-4")
        api_key = os.getenv("QDRANT_API_KEY-4")
        collection_name=os.getenv("QDRANT_COLLECTION_NAME-4")

    client = qdrant_client.QdrantClient(
        host,
        api_key=api_key
    )

    print('host ' + host)
    print('api_key ' + api_key)
    print('collection_name ' + collection_name)

    collection_config = qdrant_client.http.models.VectorParams(
        size=1536, # 768 for instructor-xl, 1536 for OpenAI
        distance= qdrant_client.http.models.Distance.COSINE
    )
    print('collection_config')
    print(recreate )
    if recreate is True:
        print('recreate')        
        print(collection_name)
        
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=collection_config
        )
        print('recreate 0')
        
    print('recreate 1')
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )

    print('after vectorstore')

    if selected_doc is not None:
        my_dict = {}
        my_dict["title"] = selected_doc
        dict_list = []
        for _ in text_chunks:
            dict_list.append(my_dict)
            print('masuk loop')

        vectorstore.add_texts(text_chunks, metadatas=dict_list)
    else:
        vectorstore.add_texts(text_chunks)
    #print(text_chunks)
def knowledge(request):

    file = FileModel.objects.last()
    
    # module_dir = os.path.dirname(__file__)
    # file_path = os.path.join(module_dir, 'static/why nation fail - labeled.txt')   #full path to text.
    
    with open(file.file.path, encoding='utf8') as f:
        # contents = f.read()
        contents = ''
        for line in f:
            contents += line.strip() + '<br>'
    if contents == '':
        context = {'knowledges': 'empty'}
    else:
        #context = {'knowledges': 'empty'}
        context = {'knowledges': contents}
    return render(request,'knowledge.html', context)


# def chat_thebot(request):
#     "' this is chat function '"
#     if not request.session.session_key:
#         request.session.create()

#     load_dotenv()
#     if request.POST.get('question'):
#         selected_vec = ''
#         question = request.POST.get('question')
#         selected_vec = docs_arr[3]
#         chat_history = request.POST.getlist('chat_history[]')
#         if len(chat_history) < 2:
#             chat_history = [] 

#         message = handle_conversation_qdrant_vector(request, question, selected_vec, chat_history)

#         print('message 3 : ' + message['answer'])
#         if message['answer'] == 'noinfo':   
#             message      = ''
#             chat_history = '' 
#             selected_vec = ''   

#             chat_history = request.POST.getlist('chat_history[]')
#             chat_chunk_size = request.POST.get('chat_chunk_size', '1000')
#             #chat_knowledge = request.POST.get('chat_knowledge', '1000')

#             print('chat chunk ' + chat_chunk_size)

#             chunk_size = int_or_0(chat_chunk_size)
#             #knowledge = int_or_0(chat_knowledge)
            
            
#             if chunk_size == 500:
#                 selected_vec = docs_arr[0]
#             elif chunk_size == 750:
#                 selected_vec = docs_arr[1]
#             else:
#                 selected_vec = docs_arr[2]

#             if len(chat_history) < 2:
#                 chat_history = []

#             message = handle_conversation_qdrant_vector(request, question, selected_vec, chat_history)
#         # message = f"You ask: {request.GET['question']} "
#         # print(message)
#     else:
#         message = {}
#         message['status'] = False
#         message['message'] = 'You ask nothing ?'
#     return HttpResponse(json.dumps(message), content_type="application/json")

def chat_thebot(request):
    "' this is chat function '"
    if not request.session.session_key:
        request.session.create()

    load_dotenv()
    if request.POST.get('question'):
        question = request.POST.get('question')
        chat_history = request.POST.getlist('chat_history[]')
        chat_chunk_size = request.POST.get('chat_chunk_size', '1000')
        # chat_knowledge = request.POST.get('chat_knowledge', '1000')
        message = run_conversation(request, question, chat_chunk_size, chat_history)       
        
        # message = f"You ask: {request.GET['question']} "
        print(message)
    else:
        message = {}
        message['status'] = False
        message['message'] = 'You ask nothing ?'
    return HttpResponse(json.dumps(message), content_type="application/json")





def get_vector_from_qdrant(query, selected_doc):
    """get all vector from qdrant"""

    print('selected_doc'+ selected_doc)
    print('docs_arr[3]'+ docs_arr[3])
    
    if selected_doc == docs_arr[0]:
        host = os.getenv("QDRANT_HOST")
        api_key = os.getenv("QDRANT_API_KEY")
        collection_name=os.getenv("QDRANT_COLLECTION_NAME")

    elif selected_doc == docs_arr[1]:
        host = os.getenv("QDRANT_HOST-2")
        api_key = os.getenv("QDRANT_API_KEY-2")
        collection_name=os.getenv("QDRANT_COLLECTION_NAME-2")

    elif selected_doc == docs_arr[2]:
        host = os.getenv("QDRANT_HOST-3")
        api_key = os.getenv("QDRANT_API_KEY-3")
        collection_name=os.getenv("QDRANT_COLLECTION_NAME-3")

    elif selected_doc == docs_arr[3]:
        host = os.getenv("QDRANT_HOST-4")
        api_key = os.getenv("QDRANT_API_KEY-4")
        collection_name=os.getenv("QDRANT_COLLECTION_NAME-4")


    print(host)
    print(api_key)
    print(collection_name)

    client = qdrant_client.QdrantClient(
        host,
        api_key=api_key
    )
    embeddings = OpenAIEmbeddings()
    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )

    # found_docs = vector_store.similarity_search(query)
    return vector_store

def handle_conversation_qdrant_vector(request, user_question, selected_doc, pre_chat_history):
    """handle user input with conversation chain"""

    vector_store = get_vector_from_qdrant(user_question, selected_doc)
    # request.session["conversation"] = get_conversation_chain(vector_store, selected_doc)
    # pre_chat_history = ['what is this about ?','This conversation appears to be about the topic of cultural evolution and its impact on different geographic regions. The speakers discuss how cultural differences can lead to one group feeling superior to another. They also briefly mention the use of AI (Artificial Intelligence) and its potential positive impact on society. Additionally, they touch on the topic of leadership and what qualities are necessary for effective leadership in a country or nation.']
    print('before get_conversation_chain')
    chain = get_conversation_chain(vector_store, selected_doc, pre_chat_history)
    tokencount = {}
    docs = []

    with get_openai_callback() as cb:
        # response = request.session["conversation"]({'question': user_question})
        response = chain({'question': user_question})

        logger.info("Q: " + response['chat_history'][len(response['chat_history'])-2].content)
        logger.info("A: " + response['chat_history'][len(response['chat_history'])-1].content)
        logger.info(cb)

        sources = response['source_documents']
        for d in sources:
            docs.append({'page_content': d.page_content, 'metadata': d.metadata})

        logger.info("D: " + json.dumps(docs))

        tokencount['token_used'] = cb.total_tokens
        tokencount['token_prompt'] = cb.prompt_tokens
        tokencount['token_completion'] = cb.completion_tokens
        tokencount['cost'] = cb.total_cost

    login_username = ''
    login_email = ''
    location = ''

    if request.user.is_authenticated:
        login_username = request.user.username
        login_email = request.user.email
    else:
        if request.POST.get('username'):
            login_username = 'guest ' + request.POST.get('username')
        if request.POST.get('email'):
            login_email = 'guest ' + request.POST.get('email')

    if request.POST.get('location'):
            location = request.POST.get('location')

    log = Chatlog(
        username = login_username,
        email = login_email,
        time = datetime.datetime.now(),
        location = location,
        question = response['chat_history'][len(response['chat_history'])-2].content,
        answer = response['chat_history'][len(response['chat_history'])-1].content,
        token_used = tokencount['token_used'],
        token_prompt = tokencount['token_prompt'],
        token_completion = tokencount['token_completion'],
        cost = tokencount['cost'],
        desc = json.dumps(docs),
        ip = get_client_ip(request),
        session_id = request.session.session_key,
        status = 1,
    )
    log.save()

    balikan = {}
    balikan['status'] = True
    balikan['message'] = 'Success'
    
    balikan['answer'] = response['answer']

    pre_chat_history.append(user_question)
    pre_chat_history.append(balikan['answer'])

    balikan['chat_history'] = pre_chat_history
    
    print(balikan)
    return balikan

def get_conversation_chain(vectorstore, selected_doc, chat_history):
    """the conversarion chain logic"""
    llm = ChatOpenAI(temperature=0.5)

    # memory = ConversationBufferMemory(
    #     memory_key='chat_history', return_messages=True)

    memory = ConversationBufferMemory(
        memory_key="chat_history", input_key='question', output_key='answer', return_messages=True)
    
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            memory.chat_memory.add_user_message(message)
        else:
            memory.chat_memory.add_ai_message(message)
    
    # "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in follow up question original language."
#     if selected_doc == docs_arr[3]:
#       custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in this sentence language "{question}" and if there is no information the "{question}" please send word notfound.
    
# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:"""  
#     else:
#         custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in this sentence language "{question}".

# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:"""
    custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in this sentence language "{question}".

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
    print('custom_template '+ custom_template)
    custom_prompt = PromptTemplate.from_template(custom_template)    
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        # retriever= retriever,
        retriever=vectorstore.as_retriever(),
        # retriever=vectorstore.as_retriever(sesarch_type="mmr"),
        memory=memory,
        max_tokens_limit=2000,
        # callbacks=[handler],
        verbose=True,
        return_source_documents=True,
        condense_question_prompt=custom_prompt
    )

    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm, chain_type="stuff", 
    #     verbose="True", 
    #     memory = memory,
    #     retriever=vectorstore.as_retriever(),
    #     return_source_documents = True
    #     )

    # no_op_chain = NoOpLLMChain()
    # conversation_chain.question_generator = no_op_chain

    # modified_template = "Use the following pieces of context to answer the users question. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{context}\nChat History:\n{chat_history}"

    # # modified_template = "Use the following pieces of context between ---------------- sign to answer the users question. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{context}\n----------------\nHere is your chat history:\n{chat_history}\nSystem: Don't use chat history. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer."

    # # modified_template = "Use the following pieces of context between [] sign to answer the users question and do not use chat history between () sign as your context. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n[\n{context}\n]\nChat history:(\n{chat_history}\n)"

    # # modified_template = "Please follow the 2 instructions below:\n1. Use the following pieces of context to answer the users question. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n2.If Assistant don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{context}\nChat History:\n{chat_history}"

    # modified_template = "Use the following pieces of context to answer the users question with the same language as the user question. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{context}\n"
    
    # system_message_prompt = SystemMessagePromptTemplate.from_template(modified_template)
    # conversation_chain.combine_docs_chain.llm_chain.prompt.messages[0] = system_message_prompt

    # # add chat_history as a variable to the llm_chain's ChatPromptTemplate object
    # conversation_chain.combine_docs_chain.llm_chain.prompt.input_variables = ['context', 'question', 'chat_history']

    return conversation_chain

def get_perhitunganpermi_chain(vectorstore, chat_history):
    """the conversarion chain logic"""
    llm = ChatOpenAI(temperature=0.5)
    print('masuk get_perhitunganpermi_chain')

    memory = ConversationBufferMemory(
    memory_key="chat_history", input_key='question', output_key='answer', return_messages=True)    
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            memory.chat_memory.add_user_message(message)
        else:
            memory.chat_memory.add_ai_message(message)
    
    # "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in follow up question original language."
    custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in this sentence language "{question}" and if there is no information about what is being asked, answer with word noinfo.
    
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
    print('custom_template : '+ custom_template)
    custom_prompt = PromptTemplate.from_template(custom_template)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        # retriever= retriever,
        retriever=vectorstore.as_retriever(),
        # retriever=vectorstore.as_retriever(sesarch_type="mmr"),
        memory=memory,
        max_tokens_limit=2000,
        # callbacks=[handler],
        verbose=True,
        return_source_documents=True,
        condense_question_prompt=custom_prompt
    )
    print('return conversation_chain')
    return conversation_chain

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def signup(request):
    if request.user.is_authenticated:
        return redirect('/')
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            login(request, user)
            return redirect('/')
        else:
            return render(request, 'signup.html', {'form': form})
    else:
        form = UserCreationForm()
        return render(request, 'signup.html', {'form': form})
    
def signin(request):
    if request.user.is_authenticated:
        return render(request, 'home.html')
    if request.method == 'POST':
        # username = request.POST['username']
        # password = request.POST['password']
        username = request.POST.get('username')
        password = request.POST.get('password')
        # print(username)
        user = authenticate(request, username=username, password=password)
        # print(user)
        if user is not None:
            login(request, user)
            return redirect('/') #profile
        else:
            msg = 'Error Login'
            form = AuthenticationForm(request.POST)
            return render(request, 'login.html', {'form': form, 'msg': msg})
    else:
        form = AuthenticationForm()
        return render(request, 'login.html', {'form': form})
  
def profile(request): 
    return render(request, 'profile.html')
   
def signout(request):
    logout(request)
    return redirect('/')

def num_tokens_from_string(string: str, encoding_name: str) -> int:    
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def getchatlog():
    result_list = list(Chatlog.objects.all())
    print(result_list)
  
    return JsonResponse(result_list, safe=False)

def getdesc_chatlog(request):
    "'get chat log chunk'"
    id_log = request.GET['id']
    chat = Chatlog.objects.filter(id=int(id_log)).first()
    
    balikan = {}
    balikan['status'] = True
    balikan['desc'] = chat.desc

    return JsonResponse(balikan, safe=False)

def get_chatlog(request):
    "'get chatlog for api'"
    page = request.POST.get('page', 1)
    per_page = request.POST.get('per_page', 10)

    check_digit = True
    testcases = [page, per_page]
    for case in testcases:
        if case.isdigit():
            print(case, " is an integer.")
        else:
            check_digit = False

    if check_digit is False:
        balikan = {}
        balikan['status'] = False
        balikan['page'] = page
        balikan['per_page'] = per_page
        balikan['data'] = {}
        balikan['message'] = 'page or per page is not integer'
        return JsonResponse(balikan, safe=False)

    log =list( Chatlog.objects.all().order_by("-time"))
    paginator = Paginator(log, per_page=int(per_page))
    requested_log = list(paginator.get_page(int(page)))

    serialized_queryset = serializers.serialize('python', requested_log)

    balikan = {}
    balikan['status'] = True
    balikan['page'] = page
    balikan['per_page'] = per_page
    balikan['num_pages'] = paginator.num_pages
    balikan['data'] = serialized_queryset

    return JsonResponse(balikan, safe=False)


# def get_parameter_calculate_premi(vehicle_type,vehicle_year,vehicle_license,sum_insured):
#     """Get the parameter to calculate premi"""
#     param_calcualtion = {
#         "vehicle_type": vehicle_type,
#         "vehicle_year": vehicle_year,
#         "vehicle_license": vehicle_license,
#         "sum_insured": sum_insured,       
#     }
#     return json.dumps(param_calcualtion)

def call_form_calculate_premi():
    """Call From Calculation Premi""" 

def call_claim():
    """Call Claim""" 

def call_Premi_Calcuation(dataPolicy):

    print(dataPolicy['vehicle_make'])
    print(dataPolicy['kategori_kendaraan'])
    print(dataPolicy['model_kendaraan'])
    print(dataPolicy['vehicle_year'])
    print(dataPolicy['vehicle_license'])
    print(dataPolicy['sum_insured'])
    print(dataPolicy['jenis_asuransi'])


    vehicle_make = dataPolicy['vehicle_make']
    vehicle_category = dataPolicy['kategori_kendaraan']
    vehicle_model = dataPolicy['model_kendaraan']
    vehicle_year = dataPolicy['vehicle_year']    
    license_plate = dataPolicy['vehicle_license']
    sum_insured = dataPolicy['sum_insured']
    jenis_asuransi = dataPolicy['jenis_asuransi'].upper()
    jenis_asuransi_desc = ''
    if jenis_asuransi =='ALL':
        jenis_asuransi_desc = 'COMPREHENSIVE'
    elif jenis_asuransi =='TLO':
        jenis_asuransi_desc = 'TLO'
    i = 0
    num = ''
    if dataPolicy['totalPerluasan'] != 0 :
        while(i < dataPolicy['totalPerluasan']):
            num = str(i)
            if i == 0:
                uw_coverage = '{   "COVERAGE_CODE" : "'+ dataPolicy['prluasan'+num] +'",    "COVERAGE_AMOUNT" : ""  }'  
            else:   
                uw_coverage = uw_coverage + ',{   "COVERAGE_CODE" : "'+ dataPolicy['prluasan'+num] +'",    "COVERAGE_AMOUNT" : ""  }'           
            i = i+1
        uw_coverage = '[ '+ uw_coverage+ ' ]'
    else:
        uw_coverage = '[ ]'
    print('uw_coverage')
    print(uw_coverage)

    url = os.getenv("URL_PREMI_CALCULATION")
    # params = {
    #             "api": "submit_simulation_4auto_or_motor",
    #             "device" : "chatbot",
    #             "hardwareId" : "9ED21E59-5DFB-4ED6-B4BC-993B443EF71A",
    #             "osVersion": "1",
    #             "username": "yuliusadityaprimandaru@gmail.com",
    #             "language": "id",
    #             "push_token": "b68c994462a88ad09f1f2910c26ad72692d5ae992eb111bc2d37fe91607a76de",
    #             "userid": "42",
    #             "token": "FqtEWKtS8Xrruk2aACbtjc3hOqLk9KvP",
    #             "uw_class": "AUTO",
    #             "vehicle_make": vehicle_make,
    #             "vehicle_category": vehicle_category,
    #             "vehicle_model" :vehicle_model,
    #             "coverage_type": "ALL",
    #             "coverage_desc": "COMPREHENSIVE",
    #             "license_plate": license_plate,
    #             "vehicle_reg_no": "12vv",
    #             "year": vehicle_year,
    #             "occupation": "PRIVATE",
    #             "usage": "NEW",
    #             "price": sum_insured,
    #             "uw_coverage":[
    #                                 {
    #                                     "COVERAGE_CODE" : "PERIL005",
    #                                     "COVERAGE_AMOUNT" : ""
    #                                 },
    #                                 {
    #                                     "COVERAGE_AMOUNT" : "",
    #                                     "COVERAGE_CODE" : "PERIL006"
    #                                 }
    #                             ]
    #         }
 

    payload = {'api': 'submit_simulation_4auto_or_motor',
                'device': 'chatbot',
                'hardwareId': '9ED21E59-5DFB-4ED6-B4BC-993B443EF71A',
                'osVersion': '1',
                'username': 'yuliusadityaprimandaru@gmail.com',
                'language': 'id',
                'push_token': 'b68c994462a88ad09f1f2910c26ad72692d5ae992eb111bc2d37fe91607a76de',
                'userid': '42',
                'token': 'FqtEWKtS8Xrruk2aACbtjc3hOqLk9KvP',
                'uw_class': 'AUTO',
                'vehicle_make': vehicle_make,
                'vehicle_category': vehicle_category,
                'vehicle_model': vehicle_model,
                'coverage_type': jenis_asuransi,
                'coverage_desc': jenis_asuransi_desc,
                'license_plate': license_plate,
                'vehicle_reg_no': '12vv',
                'year': vehicle_year,
                'occupation': 'PRIVATE',
                'usage': 'NEW',
                'price': sum_insured,
                'uw_coverage': uw_coverage
                }
    files=[

    ]
    headers = {
    'Cookie': 'ci_session=78d10ac748ff9cfc3d6afb329db0ffacec9a4716'
    }

    response = requests.request("POST", url, headers=headers, data=payload)


    print('header xxxxx')
    print(headers)
    print(payload)

    #json_params = json.dumps(params)
    #response = requests.post(url, headers=headers, data=params)
    json_data = response.json()

    return json_data


def run_conversation(request, question, chat_chunk_size, chat_history):
    # Step 1: send the conversation and available functions to GPT
    print('question 1')
    print(question)
    messages = [{"role": "user", "content": question}]
    print('question 1.2')
    # functions = [
    #     {
    #         "name": "get_parameter_calculate_premi",
    #         "description": "Get the parameter to calculation premi",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "vehicle_type": {
    #                     "type": "string",
    #                     "description": "The type of vehicle, e.g. Motor, Sedan, Truck",
    #                 },
    #                 "vehicle_year": {
    #                     "type": "string",
    #                     "description": "Vehicle Manufacturing year e.g. 2015, 2013 ",
    #                 },
    #                 "vehicle_license": {
    #                     "type": "string",
    #                     "description": "The Vehicle license  e.g. AB35950cc ",
    #                 },
    #                 "postal_code": {
    #                     "type": "string",
    #                     "description": "Postal code e.g. 55563 ",
    #                 },
    #                  "sum_insured": {
    #                     "type": "string",
    #                     "description": "Vehicle Price e.g. 150000000",
    #                 },
    #                 #"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    #             },
    #             "required": ["location"],
    #         },
    #     }
    # ]

    # functions = [
    #     {
    #         "name": "get_parameter_calculate_premi",
    #         "description": "Mengambil parameter untuk yang di butuhkan untuk menghitung premi",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "vehicle_type": {
    #                     "type": "string",
    #                     "description": "Tipe kendaraan yang akan diasuransikan contohnya: Motor, Sedan, Truck",
    #                 },
    #                 "vehicle_year": {
    #                     "type": "string",
    #                     "description": "Tahun pembuatan kendaraan contohnya: 2015,2018 ",
    #                 },
    #                 "vehicle_license": {
    #                     "type": "string",
    #                     "description": "No Polisi atau No lisensi Kendaraan contohnya: AB35950XX,B56777VG ",
    #                 },                 
    #                  "sum_insured": {
    #                     "type": "string",
    #                     "description": "Harga beli dari kendaraan yang akan diasusransikan contohnya : 150000000",
    #                 },
    #                 #"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    #             },
    #          },
    #     }
    # ]

    # print('question 2')
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo-0613",
    #     messages=messages,
    #     functions=functions,
    #     function_call="auto",  # auto is default, but we'll be explicit
    # )
    # print('question 3')

    

    functions = [
        # {
        #     "name": "call_form_calculate_premi_fire",
        #     "description": "Fungsi untuk menghitung premi asuransi kebakaran",
        #     "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "dummy_property": {
        #                         "type": "null",
        #                         }
        #                     }
        #                 }
        # },
        {
            "name": "call_form_calculate_premi",
            "description": "Fungsi untuk menghitung premi asuransi kendaraan bermotor",
            "parameters": {
                        "type": "object",
                        "properties": {
                            "dummy_property": {
                                "type": "null",
                                }
                            }
                        }
        },        
        {
            "name": "call_claim",
            "description": "Fungsi untuk melakukan claim asuransi kendaraan bermotor",
            "parameters": {
                        "type": "object",
                        "properties": {
                            "dummy_property": {
                                "type": "null",
                                }
                            }
                        }
        }
    ]

    print('question 2')
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    print('question 3')
    print(response)
    responses = response["choices"][0]["message"]
    if (responses.get("function_call")) :
        response_message = response["choices"][0]["message"]["function_call"]
        print('question 4')
        print(response_message.get("name"))
        # Step 2: check if GPT wanted to call a function
        balikan = {}      
        vehicle_make    = list(Vehicle_Master.objects.all().values('vehicle_make').distinct().order_by('vehicle_make')) 
        kode_huruf      = list(Plat_No.objects.all().values('kode_huruf').distinct().order_by('kode_huruf'))
        # Convert QuerySet to a list of dictionaries

        print('question 5')
        print(vehicle_make)
        print(kode_huruf)
        vehicle_list = []
        plat_list = []
        # plat_list = []
        i = 0
        for i in range(len(vehicle_make)):
            vehicle_list.append(vehicle_make[i]['vehicle_make']) 
            i = i+1

        j = 0
        for j in range(len(kode_huruf)):
            plat_list.append(kode_huruf[j]['kode_huruf']) 
            j = j+1
        # for item2 in plat_data:
        #     plat_list.append(item2)     
        #serialized_vehicle = serializers.serialize('python', vehicle_data)
        print('question 6')
        #print(serialized_vehicle)
        balikan['vehicle'] = vehicle_list
        balikan['plat'] = plat_list

        if response_message.get("name") == 'call_form_calculate_premi':       
            print('question 4.1')
            #request, question, chat_chunk_size, chat_history
            
            balikan['status'] = True
            balikan['message'] = 'PopUp'        
            balikan['answer'] = 'Simulasi harga premi kendaraan bermotor'
            balikan['question'] = question
            balikan['chat_chunk_size'] = chat_chunk_size
            balikan['chat_history'] = chat_history            
            print('question 4.2')               
            
        elif response_message.get("name") == 'call_claim':    
            balikan['status'] = True
            balikan['message'] = 'PopUp'        
            balikan['answer'] = 'Claim'
            balikan['question'] = question
            balikan['chat_chunk_size'] = chat_chunk_size
            balikan['chat_history'] = chat_history         
        # elif response_message.get("name") == 'call_form_calculate_premi_fire':   
        #     balikan['status'] = True
        #     balikan['message'] = 'PopUp'        
        #     balikan['answer'] = 'Simulasi harga premi kebakaran'
        #     balikan['question'] = question
        #     balikan['chat_chunk_size'] = chat_chunk_size
        #     balikan['chat_history'] = chat_history   
        else:
            chunk_size = int_or_0(chat_chunk_size)
            
            selected_vec = ''
            if chunk_size == 500:
                selected_vec = docs_arr[0]
            elif chunk_size == 750:
                selected_vec = docs_arr[1]
            else:
                selected_vec = docs_arr[2]

            if len(chat_history) < 2:
                chat_history = []

            balikan = handle_conversation_qdrant_vector(request, question, selected_vec, chat_history)
            

        
    else:
        chunk_size = int_or_0(chat_chunk_size)
        
        selected_vec = ''
        if chunk_size == 500:
            selected_vec = docs_arr[0]
        elif chunk_size == 750:
            selected_vec = docs_arr[1]
        else:
            selected_vec = docs_arr[2]

        if len(chat_history) < 2:
            chat_history = []

        balikan = handle_conversation_qdrant_vector(request, question, selected_vec, chat_history)
        print(balikan)
    return balikan 
    


def permi_calculation(request):
    print('masuk permi_calculation')
    balikan = {}


    if request.method == 'POST':        
        dataPolicy={}

        dataPolicy['vehicle_make'] = request.POST.get('merek_kendaraan')
        dataPolicy['kategori_kendaraan'] = request.POST.get('kategori_kendaraan')
        dataPolicy['model_kendaraan'] = request.POST.get('model_kendaraan')
        dataPolicy['vehicle_year'] = request.POST.get('tahun_kendaraan')
        dataPolicy['vehicle_license'] = request.POST.get('plat_no')
        dataPolicy['sum_insured'] = request.POST.get('harga_kendaraan')
        dataPolicy['jenis_asuransi'] = request.POST.get('jenis_asuransi')
        perluasanJaminan = request.POST.getlist('perluasan_jaminan[]')
        dataPolicy['prluasan1'] =''
        dataPolicy['prluasan2'] =''
        dataPolicy['prluasan3'] =''
        dataPolicy['prluasan4'] =''
        dataPolicy['prluasan5'] =''

        i = 0
        num = ''
        for i, option in enumerate(perluasanJaminan):
            num = str(i)
            dataPolicy['prluasan'+num] = option    
            print(dataPolicy['prluasan'+num])       
            i = i+1
        dataPolicy['totalPerluasan'] = i
        question = request.POST.get('question')
        chat_chunk_size = request.POST.get('chat_chunk_size')
        chat_history = request.POST.getlist('chat_history[]')
        i = 0
        num = ''
        perluasanJaminan =''
        while(i < dataPolicy['totalPerluasan']):
            num = str(i)
            perluasanJaminan = perluasanJaminan +'\n-'+ dataPolicy['prluasan'+num]          
            i = i+1
        
        
        print('nilai balikan')
        print(dataPolicy['vehicle_make'])
        print(dataPolicy['kategori_kendaraan'])
        print(dataPolicy['model_kendaraan'])
        print(dataPolicy['vehicle_year'])
        print(dataPolicy['vehicle_license'])
        print(dataPolicy['sum_insured'])
        print(dataPolicy['jenis_asuransi'])
        print(dataPolicy['totalPerluasan'])
        print('perluasanJaminan 1 : '+perluasanJaminan)
        print(chat_history)  
        if perluasanJaminan == '' or perluasanJaminan is None :
            perluasanJaminan = '-'            
        else:
            perluasanJaminan = perluasanJaminan.replace('PERIL002','TJH PIHAK III').replace('PERIL003','KERUSUHAN, PEMOGOKAN DAN HURU-HARA').replace('PERIL004','BAJINR, ANGIN TOPAN DAN BADAI').replace('PERIL005','GEMPA BUMI, LETUSAN GUNUNG BERAPI DAN TSUNAMI').replace('PERIL006','BENGKEL RESMI')
        print('perluasanJaminan 2 : '+perluasanJaminan)
        
        parsed_data     = call_Premi_Calcuation(dataPolicy)
        premi = str(re.findall(r"IDR ([\d,]+\.\d+)", parsed_data['message']))
        print(premi)
        premi = premi.replace("[", "").replace("]", "").replace("'", "")
        print(premi)
        print("parsed_data : ",parsed_data) 

        status           = parsed_data['status'] 
        TemplateAnswer = 'Dari informasi yang anda masukkan:\n' + \
                        'Merek: ' + dataPolicy['vehicle_make'] + '\n' + \
                        'Kategori: ' + dataPolicy['kategori_kendaraan'] + '\n' + \
                        'Model: ' + dataPolicy['model_kendaraan'] + '\n' + \
                        'Plat Kendaraan: ' + dataPolicy['vehicle_license'] + '\n' + \
                        'Tahun Pembuatan: ' + dataPolicy['vehicle_year'] + '\n' + \
                        'Harga Kendaraan: ' + dataPolicy['sum_insured'] + '\n' + \
                        'Perluasan Jaminan: ' + perluasanJaminan + '\n' + \
                        'Perkiraan harga premi asuransi kendaraan anda adalah sebesar Rp ' + premi
                   
        if status == 1:
            balikan['status'] = True
            balikan['message'] = 'Success'        
            balikan['answer'] = TemplateAnswer
            chat_history.append(question)
            chat_history.append(balikan['answer'])
            balikan['chat_history'] = chat_history
        else:
            
            balikan['status'] = True
            balikan['message'] = 'Error'        
            balikan['answer'] = parsed_data['message']
            chat_history.append(question)
            chat_history.append(balikan['answer'])
            balikan['chat_history'] = chat_history
            print('respon call API')

    else:
        balikan['status'] = True
        balikan['message'] = 'Error'        
        balikan['answer'] = 'Invalid request method'
        chat_history.append(question)
        chat_history.append(balikan['answer'])
        balikan['chat_history'] = chat_history
    return HttpResponse(json.dumps(balikan), content_type="application/json")

def get_dropdown_category(request):
    print('get dropdown vehicle_make')
    vehicleMake = request.GET.get('merekKendaraan')
    print(vehicleMake)
    vehicle_category = Vehicle_Master.objects.filter(vehicle_make=vehicleMake).values('vehicle_category').distinct().order_by('vehicle_category')
    print('vehicle_category')
    print(vehicle_category)
    return JsonResponse({
        'vehicle_category': list(vehicle_category)
    })

def get_dropdown_model(request):
    print('get dropdown vehicle_kategori')
    vehicleKategori = request.GET.get('kategoriKendaraan')
    print(vehicleKategori)
    vehicle_model = Vehicle_Master.objects.filter(vehicle_category=vehicleKategori).values('vehicle_model').distinct().order_by('vehicle_model')
    print('vehicle_model')
    print(vehicle_model)
    return JsonResponse({
        'vehicle_model': list(vehicle_model)
    }) 
    
class OrderListJson(BaseDatatableView):
    # The model we're going to show
    model = Chatlog

    # define the columns that will be returned
    # columns = ['id', 'username', 'email', 'time', 'question', 'answer',
    #            'location','token_prompt','token_completion','token_used','cost','desc','status']
    
    columns = ['id', 'time', 'username', 'location', 'token_used', 'question', 'desc']

    # define column names that will be used in sorting
    # order is important and should be same as order of columns
    # displayed by datatables. For non-sortable columns use empty
    # value like ''
    order_columns = ['id', 'time', 'username', 'token_used', 'question']

    # set max limit of records returned, this is used to protect our site if someone tries to attack our site
    # and make it return huge amount of data
    max_display_length = 500

    def render_column(self, row, column):
        # We want to render user as a custom column
        # if column == 'username':
        #     return row.username + '<br>' +  row.email
        if column == 'time':
            # escape HTML for security reasons
            return escape('{0}'.format(row.time.strftime("%d-%m-%Y %H:%M:%S")))
        if column == 'token_used':
            return 'token : ' + str(row.token_used) + '<br> $' +  str(row.cost)
        if column == 'location':
            return 'IP : ' + str(row.ip) + '<br>' +  str(row.location)
        if column == 'question':
            return row.question + '<br><b>' +  row.answer + '</b>'
        if column == 'desc':
            return '<button type="button" class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#exampleModalLong" onclick="get_modal_data(this, '+str(row.id)+')">Chunks</button>'
        else:
            return super(OrderListJson, self).render_column(row, column)

   
class NoOpLLMChain(LLMChain):
   """No-op LLM chain."""

   def __init__(self):
       """Initialize."""
       super().__init__(llm=ChatOpenAI(), prompt=PromptTemplate(template="", input_variables=[]))

   def run(self, question: str, *args, **kwargs) -> str:
       return question