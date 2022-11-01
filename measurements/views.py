from cProfile import label
from cmath import log, log10
from io import BytesIO
from telnetlib import PRAGMA_HEARTBEAT
from turtle import color
from unicodedata import name
from django.shortcuts import render, redirect
from numpy import dtype, integer, size
import sqlalchemy
from experiments.models import Experiment
import plotly.express as px
from measurements.tables import MonomerTable
from .forms import AddFileForm
from .models import Measurement, Data, Monomer, Experiment
from django.contrib.auth.decorators import login_required
from plotly.offline import plot
import plotly.graph_objects as go
import sqlite3
import mysql.connector
import datetime
import pandas
import pymysql as mdb
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import base64
from io import BytesIO
import plotly as p
import plotly.graph_objects as go
import numpy as np
import mpl_toolkits.mplot3d
from django.views.decorators.csrf import csrf_exempt
import math


def csv_to_db(file, pk):

    data = pandas.read_csv(file.file, encoding='UTF-8')

    data_conv = data[['conversion', 'tres']]
    data_conv['tres'] = data_conv.apply(
        lambda row: datetime.timedelta(minutes=row.tres).total_seconds(), axis=1)
    data_conv.rename(columns={'conversion': 'result',
                     'tres': 'res_time'}, inplace=True)
    data_conv['measurement_id'] = pk

    print(data_conv)

    con = sqlalchemy.create_engine("mysql+mysqldb://root@localhost/chemistry")
    con = con.connect()

    data_conv.to_sql('measurements_data', con,
                     if_exists='append', index=False, method='multi')


@login_required
def monomer_kinetics(request):
    monomers = MonomerTable(Monomer.objects.all())

    context = {
        'monomer': monomers,
    }

    return render(request, "measurements/show3Dplot.html", context)


@login_required
def upload_file(request, pk):
    if request.method == 'POST':
        form = AddFileForm(request.POST, request.FILES, pk=pk)
        if form.is_valid():
            m = form.save()
            csv_to_db(m.file, m.pk)
        else:
            print(form.errors)

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
    print(df)
    max = df['res_time'].max() + df['res_time'].min()
    print(max)
    max = int(float(max))
    max -= max % -100
    print(max)

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


def get_CTA_reaction_data(request, name):
    df = pandas.DataFrame(
        list(Experiment.objects.filter(monomer=name).values('id')))
    list_ids = df.values.tolist()

    wanted_experiment_ids_list = [
        item for sublist in list_ids for item in sublist]

    df_temp = pandas.DataFrame(
        list(Experiment.objects.filter(monomer=name).values('id', 'temperature', 'CTA', 'cx_ratio')))
    list_temp = df_temp.values.tolist()

    df_measurements = pandas.DataFrame(
        list(Measurement.objects.values('experiment_id', 'id')))
    df_measure = df_measurements[df_measurements['experiment_id'].isin(
        wanted_experiment_ids_list)]
    list_measure = df_measure.values.tolist()

    measurement_ids_for_data = []

    for item in df_measure['id']:
        measurement_ids_for_data.append(item)

    df_data_dataframe = pandas.DataFrame(
        list(Data.objects.values('measurement_id', 'res_time', 'result')))
    df_data = df_data_dataframe[df_data_dataframe['measurement_id'].isin(
        measurement_ids_for_data)]
    list_data = df_data.values.tolist()
    return join_monomer_temperature(list_data, list_measure, list_temp)


def join_monomer_temperature(list_data, list_measure, list_temp):
    for i in range(len(list_data)):
        for x in range(len(list_measure)):
            if list_data[i][0] == list_measure[x][1]:
                list_data[i][0] = list_measure[x][0]
                break

    for i in range(len(list_temp)):
        for x in range(len(list_data)):
            if list_data[x][0] == list_temp[i][0]:
                list_data[x][0] = list_temp[i][1]
                list_data[x].append(list_temp[i][2])
                list_data[x].append(list_temp[i][3])
    return list_data


def get_axis(list_data):
    columns = list(zip(*list_data))
    x = list(columns[0])
    y = list(columns[1])
    z = list(columns[2])
    CTA = list(columns[3])
    cx_ratio = list(columns[4])

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)
    cx_ratio = np.array(cx_ratio, dtype=float)
    return x, y, z, cx_ratio, CTA


def modify_axis(data, axis_label, axis_input):
    try:
        for i in range(len(data[axis_label])):
            new_formula = str(axis_input).replace(
                "x", str(data[axis_label][i]))
            new_value = eval(new_formula)
            data[axis_label][i] = new_value.real
    except:
        print("not valid")
        pass


def plot_3d_graph(request, name):

    list_data = get_CTA_reaction_data(request, name)
    x, y, z, cx_ratio, CTA = get_axis(list_data)

    data = {
        "temperature(C)": x,
        "residence_time(s)": y,
        "conversion%": z,
        "Chain Transfer Agent": CTA,
        "cx ratio": cx_ratio,

    }
    df = pandas.DataFrame(data)

    fig = px.scatter_3d(df, x='temperature(C)',
                        y='residence_time(s)', z='conversion%', color="Chain Transfer Agent", symbol="cx ratio")
    fig.update_traces(marker=dict(size=5),
                      selector=dict(mode='markers'))
    return fig.to_html()


@csrf_exempt
@login_required
def view_3d_graph(request, name):

    axis = [
        "temperature(C)",
        "residence_time(s)",
        "conversion%"

    ]

    if request.method == 'POST':
        left_axis = request.POST.get("axis_left")
        left_input = request.POST.get("left-input")
        middle_axis = request.POST.get("axis_middle")
        middle_input = request.POST.get("middle-input")
        right_axis = request.POST.get("axis_right")
        right_input = request.POST.get("right-input")
        data = modify_axis(data, left_axis, left_input)
        data = modify_axis(data, middle_axis, middle_input)
        data = modify_axis(data, right_axis, right_input)

    plot_3d = plot_3d_graph(request, name)
    context = {
        'plot_3d_graph': plot_3d,
        'axis': axis,
        'name': name
    }

    return render(request, 'measurements/3D_graph.html', context)


@csrf_exempt
@login_required
def view_3d_kinetic_graph(request, name):

    three_d_graph = plot_3d_graph(request, name)
    list_data = get_CTA_reaction_data(request, name)
    temperature, y, z, cx_ratio, CTA = get_axis(list_data)

    CTA_list = set(CTA)
    temperature_list = set(temperature)

    reaction_orders = [
        "1st Order",
        "2nd Order",
        "3rd Order"
    ]
    cx_ratio = set(cx_ratio)
    context = {
        'temperature_list': temperature_list,
        'CTA_list': CTA_list,
        'name': name,
        'reaction_orders': reaction_orders,
        'plot_3d_graph': three_d_graph,
        'cx_ratio': cx_ratio}

    return render(request, 'measurements/kinetic_view.html', context)
