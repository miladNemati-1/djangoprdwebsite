import imp
from django.shortcuts import render, redirect
from numpy import product
from .forms import AddFileForm
from .models import Measurement, Data
from django.contrib.auth.decorators import login_required
from plotly.offline import plot
import plotly.graph_objects as go
import sqlite3
import datetime
import pandas
import csv
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage
from rest_framework import serializers, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt


def csv_to_db(file, pk):

    data = pandas.read_csv(file.file, encoding='UTF-8')

    data_conv = data[['conversion', 'tres']]
    data_conv['tres'] = data_conv.apply(
        lambda row: datetime.timedelta(minutes=row.tres).total_seconds(), axis=1)
    data_conv.rename(columns={'conversion': 'result',
                     'tres': 'res_time'}, inplace=True)
    data_conv['measurement_id'] = pk

    print(data_conv)

    con = sqlite3.connect("db.sqlite3")

    data_conv.to_sql('measurements_data', con,
                     if_exists='append', index=False, method='multi')

# @login_required


@action(detail='False', methods='POST')
@csrf_exempt
def upload_file(request, pk):

    # if request.method == 'POST':
    #     form = AddFileForm(request.POST, request.FILES, pk=pk)
    #     if form.is_valid():
    #         m = form.save()
    file = request.FILES['file']
    # form = AddFileForm(request.POST, request.FILES, pk=pk)
    # form.save()
    csv_to_db(file, pk)
    # else:
    #     print(form.errors)

    return redirect('experiment_detail', pk=pk)


@login_required
def delete_file(request, pk, path):
    if request.method == 'POST':
        file = Measurement.objects.get(pk=pk)
        file.delete()

    return redirect('experiment_detail', pk=path)


@login_required
def view_graph(request, pk):

    df = pandas.DataFrame(
        list(Data.objects.filter(measurement_id=pk).values()))
    name = Measurement.objects.get(id=pk).experiment.name

    graph_conv = go.Scatter(x=df.res_time, y=df.result, mode='markers')

    # pad for centering and round to nearest 100
    max = df['res_time'].max() + df['res_time'].min()
    max -= max % -100

    layout_conv = {
        'title': name + ": conversion",
        'xaxis_title': 't_res (s)',
        'xaxis_range': [0, max],
        'yaxis_title': 'conversion',
        'height': 630,
        'width': 840,
        'paper_bgcolor': 'rgba(0,0,0,0)',
    }

    plot_conv = plot(
        {'data': graph_conv, 'layout': layout_conv}, output_type='div')

    context = {
        'plot_conv': plot_conv,
    }

    return render(request, "measurements/view_graph.html", context)
