from cProfile import label
from cmath import e, log, log10
from io import BytesIO
from re import X
from telnetlib import PRAGMA_HEARTBEAT
from turtle import color
from unicodedata import name
from django.shortcuts import render, redirect
from numpy import dtype, integer, size
from regex import E
import sqlalchemy
from experiments.models import Experiment
import plotly.express as px
from measurements.tables import MonomerTable
from .forms import AddFileForm
from .models import Measurement, Data, Monomer, Experiment, cta
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
from scipy import stats
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.utils import Bunch
import sklearn
from sklearn.model_selection import train_test_split
# from .descision_tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import threading


@csrf_exempt
def get_experiment_rate_values(df_measurements):

    measurement_id_rate_arr = []

    measurement_ids = set(list(df_measurements['measurement_id']))
    for measurement_id in measurement_ids:
        list_of_id_values = pandas.DataFrame(
            list(Data.objects.filter(measurement_id=measurement_id).values()))
        results_floats = [(1-(log(float(1*x), e))).real
                          for x in list_of_id_values['result']]
        res_time_floats = [float(x) for x in list_of_id_values['res_time']]

        k = stats.linregress(
            results_floats, res_time_floats)[0]
        measurement_id_rate_arr.append([measurement_id, k])
    rate_measurement_id_df = pandas.DataFrame(
        measurement_id_rate_arr, columns=("measurement_id", "rate"))
    return rate_measurement_id_df


def create_descision_tree_model():
    return


def modify_list(expression, list):
    for i in range(len(list)):
        new_formula = str(expression).replace(
            "x", str(list[i]))
        try:
            list[i] = eval(new_formula)
        except:
            pass
    return list


def clean_data_frame(df):
    df.rename(columns={
        "monomer_name": "Monomer", 'cta_name': 'Chain Transfer Agent', 'res_time': 'Residence Time (min)', 'temperature': 'Temperature (C)', "result": "Conversion"}, inplace=True)
    df['Residence Time (min)'] = modify_list(
        "round((float(x)/60),3)", list(df['Residence Time (min)']))
    df['Conversion'] = modify_list(
        "round(x,3)", list(df['Conversion']))

    return df


@ csrf_exempt
@ login_required
def all_visualisations(request):
    df = get_all_res_time_conversion_data()
    df = clean_data_frame(df)
    x = 'Monomer'
    y = 'Residence Time (min)'
    z = 'Conversion'
    color = "Chain Transfer Agent"
    symbol = "cx_ratio"
    if request.method == "POST":
        x = request.POST.get("choose_x")
        y = request.POST.get("choose_y")
        z = request.POST.get("choose_z")
        x = request.POST.get("choose_x")
        color = request.POST.get("choose_colour")
        symbol = request.POST.get("choose_marker")
        x_input = request.POST.get("x-input")
        y_input = request.POST.get("y-input")
        z_input = request.POST.get("z-choose")

        df = df.dropna()
        df.to_csv("/Users/miladnemati/Desktop/file.csv")
        df = modify_axis_all_visualisations(df, x, x_input)
        df = modify_axis_all_visualisations(df, y, y_input)
        df = modify_axis_all_visualisations(df, z, z_input)

    three_d_graph = px.scatter_3d(df, x,
                                  y, z, color, symbol)
    three_d_graph = three_d_graph.to_html()

    column_names = list(df.head())

    context = {

        'plot_3d_graph': three_d_graph,
        'column_names': column_names

    }
    return render(request, "measurements/all_visualisations.html", context)


def get_all_data():
    monomer_info_df = pandas.DataFrame(
        list(Monomer.objects.values()))

    cta_info_df = pandas.DataFrame(
        list(cta.objects.values()))

    experiment_df = pandas.DataFrame(
        list(Experiment.objects.values()))

    df_measurements = pandas.DataFrame(
        list(Measurement.objects.values()))

    df_data_measurements = pandas.DataFrame(
        list(Data.objects.values()))
    df_measurement_rate = get_experiment_rate_values(df_data_measurements)

    df_measurements_measurement_rate_join = df_measurements.set_index('id').join(
        df_measurement_rate.set_index('measurement_id'))

    df_experiments_measurement_rate_join = experiment_df.set_index('id').join(
        df_measurements_measurement_rate_join.set_index('experiment_id'))

    df_experiments_monomer_join = df_experiments_measurement_rate_join.set_index('monomer_id').join(
        monomer_info_df.set_index('id').add_prefix('monomer_'))
    df_experiments_cta_join = df_experiments_monomer_join.set_index('cta_id').join(
        cta_info_df.set_index('id').add_prefix('cta_'))

    df_experiments_monomer_join.to_csv(
        "/Users/miladnemati/Desktop/df_experiments_monomer_join.csv")

    return df_experiments_cta_join


@ csrf_exempt
@ login_required
def monomer_models(request):

    df_experiments_cta_join = get_all_data()

    df_experiments_cta_join.dropna(subset=['rate'], inplace=True)

    data_target = df_experiments_cta_join[['temperature',	'cta_concentration', 'monomer_Mw', 'monomer_density_g_per_ml', 'monomer_boiling_point_celsius', 'monomer_vapour_pressure_kPa',
                                           'monomer_viscosity_cP', 'monomer_c_number', 'cta_Mw_cta', 'cta_density_g_per_ml_cta', 'cta_reflective_index_cta', 'cta_boiling_point_c_cta', 'cta_c_number_cta', 'rate']]
    data_target.to_csv("/Users/miladnemati/Desktop/model_features.csv")

    X = data_target.iloc[:, :-1].values
    Y = data_target.iloc[:, -1].values.reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=.2, random_state=41)
    regressor = DecisionTreeRegressor(min_samples_split=3)

    kinetics_model = regressor.fit(X_train, Y_train)
    Y_pred = regressor.predict(X_test)
    tree_text_represenation = tree.export_text(kinetics_model)
    predicted_k = "Predicted Rate"
    squared_error = "Mean Square Error"

    error_tree_meansquared = np.sqrt(mean_squared_error(Y_test, Y_pred))

    if request.method == 'POST':
        temp = request.POST.get("r_temp")
        cx_cm = request.POST.get("r_cx_cm")
        monomer_mw = request.POST.get("monomer_mw")
        monomer_d = request.POST.get("monomer_d")
        monomer_vp = request.POST.get("monomer_vp")
        monomer_v = request.POST.get("monomer_v")
        monomer_nc = request.POST.get("monomer_nc")
        monomer_bp = request.POST.get("monomer_bp")
        cta_mw = request.POST.get("cta_mw")
        cta_d = request.POST.get("cta_d")
        cta_ri = request.POST.get("cta_ri")
        cta_bp = request.POST.get("cta_bp")
        cta_nc = request.POST.get("cta_bp")
        try:
            prediction_input = [[temp, cx_cm, monomer_mw, monomer_d, monomer_bp, monomer_vp, monomer_v, monomer_nc, cta_mw, cta_d,
                                 cta_ri, cta_bp, cta_nc]]
            prediction = regressor.predict(prediction_input)
            predicted_k = prediction[0]
            squared_error = error_tree_meansquared
        except:
            pass

    context = {
        "predicted_k": predicted_k,
        "squared_error": squared_error

    }

    # print(error_tree_meansquared)
    return render(request, "measurements/models_home.html", context)


def csv_to_db(file, pk):

    data = pandas.read_csv(file.file, encoding='UTF-8')

    data_conv = data[['conversion', 'tres']]
    data_conv['tres'] = data_conv.apply(
        lambda row: datetime.timedelta(minutes=row.tres).total_seconds(), axis=1)
    data_conv.rename(columns={'conversion': 'result',
                     'tres': 'res_time'}, inplace=True)
    data_conv['measurement_id'] = pk

    con = sqlalchemy.create_engine("mysql+mysqldb://root@localhost/chemistry")
    con = con.connect()

    data_conv.to_sql('measurements_data', con,
                     if_exists='append', index=False, method='multi')


@ login_required
def monomer_kinetics(request):
    monomers = MonomerTable(Monomer.objects.all())

    context = {
        'monomer': monomers,
    }

    return render(request, "measurements/show3Dplot.html", context)


@ login_required
def upload_file(request, pk):
    if request.method == 'POST':
        form = AddFileForm(request.POST, request.FILES, pk=pk)
        if form.is_valid():
            m = form.save()
            csv_to_db(m.file, m.pk)
        else:
            print(form.errors)

    return redirect('experiment_detail', pk=pk)


@ login_required
def delete_file(request, pk, path):
    if request.method == 'POST':
        file = Measurement.objects.get(pk=pk)
        file.delete()

    return redirect('experiment_detail', pk=path)


@ login_required
def view_graph(request, pk):

    df = pandas.DataFrame(
        list(Data.objects.filter(measurement_id=pk).values()))
    name = Measurement.objects.get(id=pk).experiment.name

    graph_conv = go.Scatter(x=df.res_time, y=df.result, mode='markers')

    # pad for centering and round to nearest 100
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


def get_all_res_time_conversion_data():
    df_monomer = pandas.DataFrame(
        list(Monomer.objects.values()))
    df_experiment = pandas.DataFrame(
        list(Experiment.objects.values()))
    df_cta = pandas.DataFrame(
        list(cta.objects.values()))
    monomer_experiment_merge = pandas.merge(
        df_experiment, df_monomer, left_on='monomer_id', right_on='id', how='left')
    cta_monomer_experiment_merge = pandas.merge(
        monomer_experiment_merge, df_cta, left_on='cta_id', right_on='id', how='left')

    final_data = cta_monomer_experiment_merge[['id_x',
                                               'name_y', 'temperature', 'name', 'cta_concentration']]
    df_measurements = pandas.DataFrame(
        list(Measurement.objects.values('experiment_id', 'id')))

    measure_experiment_final_data = pandas.merge(
        final_data, df_measurements, left_on='id_x', right_on='experiment_id', how='left')

    df_data_dataframe = pandas.DataFrame(
        list(Data.objects.values('measurement_id', 'res_time', 'result')))

    finale_merged_data = pandas.merge(
        measure_experiment_final_data, df_data_dataframe, left_on='id', right_on='measurement_id', how='left')
    finale_merged_data.rename(columns={
        "name_y": "monomer_name", 'name': 'cta_name', 'cta_concentration': 'cx_ratio'}, inplace=True)

    finale_merged_data = finale_merged_data[[
        "monomer_name", 'cta_name', 'cx_ratio', 'res_time', 'result', 'temperature']]
    finale_merged_data.to_csv("/Users/miladnemati/Desktop/finale.csv")
    return finale_merged_data


def get_CTA_reaction_data(request, name):
    finale_merged_data = get_all_res_time_conversion_data()

    filtered_data_experiments = finale_merged_data.loc[finale_merged_data['monomer_name'] == name]

    filtered_data_experiments.to_csv("/Users/miladnemati/Desktop/finale.csv")

    return filtered_data_experiments


def get_axis(list_data):

    x = list(list_data['temperature'])
    y = list(list_data['res_time'])
    z = list(list_data['result'])
    CTA = list(list_data['cta_name'])
    cx_ratio = list(list_data['cx_ratio'])

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)
    cx_ratio = np.array(cx_ratio, dtype=float)
    return x, y, z, cx_ratio, CTA


def modify_axis_all_visualisations(data, axis_label, axis_input):
    new_data = list(data[axis_label])
    try:

        for i in range(len(new_data)):

            new_formula = str(axis_input).replace(
                "x", str(new_data[i]))
            new_data[i] = eval(new_formula).real
            print(i)

        data[axis_label] = new_data
    except:
        print("Not a valid visualisation formula")
        pass

    return data


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
    return data


def get_axis_data(list_data):
    x, y, z, cx_ratio, CTA = get_axis(list_data)

    data = {
        "temperature(C)": x,
        "residence_time(s)": y,
        "conversion%": z,
        "Chain Transfer Agent": CTA,
        "cx ratio": cx_ratio,

    }
    return data


def plot_3d_graph(request, name, df):

    fig = px.scatter_3d(df, x='temperature(C)',
                        y='residence_time(s)', z='conversion%', color="Chain Transfer Agent", symbol="cx ratio")
    fig.update_traces(marker=dict(size=5),
                      selector=dict(mode='markers'))
    return fig.to_html()


@ csrf_exempt
@ login_required
def view_3d_graph(request, name):
    list_data = get_CTA_reaction_data(request, name)
    data = get_axis_data(list_data)
    df = pandas.DataFrame(data)
    df = df.dropna()

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
        df = modify_axis_all_visualisations(df, left_axis, left_input)
        df = modify_axis_all_visualisations(df, middle_axis, middle_input)
        df = modify_axis_all_visualisations(df, right_axis, right_input)
    print(df)
    plot_3d = plot_3d_graph(request, name, df)
    context = {
        'plot_3d_graph': plot_3d,
        'axis': axis,
        'name': name

    }

    return render(request, 'measurements/3D_graph.html', context)


@ csrf_exempt
@ login_required
def view_3d_kinetic_graph(request, name):

    list_data = get_CTA_reaction_data(request, name)
    temperature, y, z, cx_ratio, CTA = get_axis(list_data)
    data = get_axis_data(list_data)
    df = pandas.DataFrame(data)
    k = "rate constant"

    CTA_list = set(CTA)
    temperature_list = set(temperature)

    reaction_orders = [
        "1st Order",
        "2nd Order",
        "3rd Order"
    ]
    cx_ratio = set(cx_ratio)

    if request.method == 'POST':
        CTA_chosen = request.POST.get("CTA")
        Temperature_chosen = request.POST.get("temperature")
        cx_ratio_chosen = request.POST.get("cx_ratio")
        order_chosen = request.POST.get("order")

        df = df.loc[(df['temperature(C)'] == float(Temperature_chosen)) & (
            df['Chain Transfer Agent'] == CTA_chosen) & (
            df['cx ratio'] == float(cx_ratio_chosen))]
        df.to_csv("/Users/miladnemati/Desktop/modification_view.csv")
        try:
            k = determine_1st_order_rate_constant(df)
        except:
            pass

    three_d_graph = plot_3d_graph(request, name, df)
    context = {
        'temperature_list': temperature_list,
        'CTA_list': CTA_list,
        'name': name,
        'reaction_orders': reaction_orders,
        'plot_3d_graph': three_d_graph,
        'cx_ratio': cx_ratio,
        'k': k
    }

    return render(request, 'measurements/kinetic_view.html', context)


def determine_1st_order_rate_constant(df):
    t = df['residence_time(s)']
    conc_M_M0_conversion = df['conversion%']
    monomer_remaining = []
    # get from db
    initial_monomer_concentration = 1
    for conversion_percent in conc_M_M0_conversion:
        monomer_remaining.append(log(
            ((1-(conversion_percent * initial_monomer_concentration)) / initial_monomer_concentration), e).real)

    k = stats.linregress(monomer_remaining, t)[0]
    return k
