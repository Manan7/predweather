from django.db import models
from django import forms


# Create your models here.
class UserForm(forms.Form):
		# date = forms.DateField(label='Select the date:', widget=forms.widgets.DateInput(attrs={'type': 'date'}))
		# date = forms.DateField(label='Select the date :P', input_formats=['%d/%m/%Y %H:%M'], widget=forms.widgets.DateInput(attrs={'id': "id_date"}))
		date = forms.DateField(label='Date:')
		#widget=forms.SelectDateWidget)