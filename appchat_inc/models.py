from django.db import models

# Create your models here.
class Chatlog(models.Model):
  username = models.CharField(max_length=255)
  email = models.CharField(max_length=255)
  time = models.DateTimeField()
  location = models.CharField(max_length=255)
  question = models.TextField()
  answer = models.TextField()
  token_used = models.IntegerField()
  token_prompt = models.IntegerField()
  token_completion = models.IntegerField()
  cost = models.DecimalField(max_digits = 19,decimal_places = 10)
  desc = models.TextField()
  ip = models.CharField(max_length=255, default=None, blank=True, null=True)
  session_id = models.CharField(max_length=255, default=None, blank=True, null=True)
  status = models.SmallIntegerField()


class FileModel(models.Model):
    file = models.FileField(upload_to='uploaded_files/', default='')
    created = models.DateTimeField(auto_now=True)
    recreate = models.BooleanField(default=True)
    model_type = models.CharField(max_length=100, default=None, blank=True, null=True)

class Vehicle_Master(models.Model):
  vehicle_make = models.CharField(max_length=20,default=None, blank=True, null=True)
  vehicle_category = models.CharField(max_length=100,default=None, blank=True, null=True)
  vehicle_model = models.CharField(max_length=60,default=None, blank=True, null=True)
  vehicle_year = models.CharField(max_length=4,default=None, blank=True, null=True)
  vehicle_assembled_by = models.CharField(max_length=40,default=None, blank=True, null=True)
  vehicle_cylinder = models.IntegerField(default=None, blank=True, null=True)
  vehicle_usage = models.CharField(max_length=2,default=None, blank=True, null=True)
  vehicle_class = models.CharField(max_length=10,default=None, blank=True, null=True)
  vehicle_door = models.IntegerField(default=None, blank=True, null=True)
  vehicle_body_type = models.CharField(max_length=50,default=None, blank=True, null=True)
  vehicle_transmition = models.CharField(max_length=2,default=None, blank=True, null=True)
  vehicle_seat = models.IntegerField(default=None, blank=True, null=True)
  vehicle_fuel_type = models.CharField(max_length=10,default=None, blank=True, null=True)
  vehicle_weight = models.IntegerField(default=None, blank=True, null=True)
  vehicle_weight_unit_of_measure = models.CharField(max_length=8,default=None, blank=True, null=True)
  vehicle_engine_cpacity = models.CharField(max_length=10,default=None, blank=True, null=True)
  vehicle_tariff = models.IntegerField(default=None, blank=True, null=True)
  remarks = models.CharField(max_length=1024,default=None, blank=True, null=True) 
  mod_date = models.DateTimeField(auto_now_add=True, blank=True)
  user_id = models.CharField(max_length=50,default=None, blank=True, null=True)
  isdiscontinue = models.CharField(max_length=1,default=None, blank=True, null=True)
  mapping_code = models.CharField(max_length=11,default=None, blank=True, null=True)
  vehicle_classification = models.CharField(max_length=10,default=None, blank=True, null=True)
  vehicle_classif_desc = models.CharField(max_length=70,default=None, blank=True, null=True) 

class Plat_No(models.Model):
  kode_huruf = models.CharField(max_length=3,default=None, blank=True, null=True)
  daerah = models.CharField(max_length=250,default=None, blank=True, null=True)

# class LocationModel(models.Model):
#     branch_code = models.CharField(max_length=2)
#     postal_code = models.CharField(max_length=10)
#     province = models.CharField(max_length=50)

   
