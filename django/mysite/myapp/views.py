from django.shortcuts import render
from myapp import weatherpredd as w
from myapp import humidity as h
from .models import UserForm
from myapp import pressure as p

# Create your views here.
def index(request):
    
    
    form= UserForm(request.POST or None)
    if form.is_valid() and request.method == 'POST':
        date= form.cleaned_data.get("date")
    else:
        date = None

    # data = {
    
    ar = w.getArimaResult()
    today = ar[0]
    today_1 = ar[8]
    coming_week = ar[1:7]
    coming__week = ar[9:15]
    # }
    ar_2 = h.getArimaResult2()
    today_2 = ar_2[0:]
    coming_week_2 = ar_2[1:7]
    coming__week_2 = ar_2[9:15]

    ar_3 = p.getArimaResult3()
    today_3 = ar_3[0]
    today__3 = ar_3[8]

    return render(request, 'index2.html', {'today' : today, 'coming_week': coming_week,'today_2' : today_2, 'coming_week_2': coming_week_2,'today_3' : today_3,'coming__week': coming__week,'today_1': today_1,'coming__week_2': coming__week_2,'today__3': today__3})