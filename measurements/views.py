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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import threading
import statsmodels


def get_all_rate_data():
    df_experiments_cta_join = get_all_cleaned_res_time_conversion_data()

    df_measurement_rate = add_experiment_rate_values_column(
        df_experiments_cta_join)

    df_measurement_rate = df_measurement_rate.set_index('rate_measurement_join_column').join(
        df_experiments_cta_join.set_index('rate_measurement_join_column'))

    return df_measurement_rate


def get_all_cleaned_res_time_conversion_data():
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

    df_data_measurements['rate_measurement_join_column'] = df_data_measurements['measurement_id']

    df_measurements_measurement_rate_join = df_measurements.set_index('id').join(
        df_data_measurements.set_index('measurement_id'))

    df_experiments_measurement_rate_join = experiment_df.set_index('id').join(
        df_measurements_measurement_rate_join.set_index('experiment_id'))

    df_experiments_monomer_join = df_experiments_measurement_rate_join.set_index('monomer_id').join(
        monomer_info_df.set_index('id').add_prefix('monomer_'))
    df_experiments_cta_join = df_experiments_monomer_join.set_index('cta_id').join(
        cta_info_df.set_index('id').add_prefix('cta_'))

    # clean data using data jump removal method
    clean_timesweep_data(df_experiments_cta_join)
    df_experiments_cta_join.replace('', np.nan, inplace=True)
    df_experiments_cta_join.dropna(inplace=True)

    return df_experiments_cta_join


def determine_rate_of_data_subset(data_subset, unique_measurement_id):

    results_floats = [(1-(log(float(1*x), e))).real
                      for x in data_subset['result']]
    res_time_floats = [float(x) for x in data_subset['res_time']]
    results_floats.insert(0, 0)
    res_time_floats.insert(0, 0)

    k = stats.linregress(
        results_floats, res_time_floats)[0]
    return [unique_measurement_id, k]


@csrf_exempt
def add_experiment_rate_values_column(df_measurements: pandas.DataFrame):
    measurement_id_rate_arr = []
    unique_data_set = set(
        list(df_measurements['rate_measurement_join_column']))
    for unique_measurement_id in unique_data_set:
        data_subset = pandas.DataFrame(
            df_measurements[df_measurements["rate_measurement_join_column"] == unique_measurement_id])
        measurement_id_rate_arr.append(
            determine_rate_of_data_subset(data_subset, unique_measurement_id))

    rate_measurement_id_df = pandas.DataFrame(
        measurement_id_rate_arr, columns=("rate_measurement_join_column", "rate"))

    return rate_measurement_id_df


def remove_data_jumps(df_time_result):
    pointer = 0
    column = df_time_result['result']
    for i in range(len(column)-1):
        next_value = float(column.iloc[i+1])
        current_pointer_value = float(column.iloc[pointer])
        if current_pointer_value + (0.05*(current_pointer_value)) < next_value or (current_pointer_value) > next_value + (0.05*(next_value)):
            column.iloc[i+1] = ''
        elif next_value < 0:
            column.iloc[i+1] = ''
        else:
            pointer = i+1
    return df_time_result


def clean_timesweep_data(monomer_conversion: pandas.DataFrame):
    df = monomer_conversion[['monomer_name',
                             'cta_name', 'cta_concentration', 'temperature']]
    list_df = df.values.tolist()
    unique_rows = np.unique(list_df, axis=0)

    for row in unique_rows:
        data_slice = monomer_conversion.loc[(
            monomer_conversion['monomer_name'] == row[0]) & (monomer_conversion['cta_name']
                                                             == row[1]) & (monomer_conversion['cta_concentration'] == row[2]) & (monomer_conversion['temperature'] == row[3])]

        monomer_conversion.loc[(
            monomer_conversion['monomer_name'] == row[0]) & (monomer_conversion['cta_name']
                                                             == row[1]) & (monomer_conversion['cta_concentration'] == row[2]) & (monomer_conversion['temperature'] == row[3])] = remove_data_jumps(
            data_slice)

    return


def clean_data_frame(df):
    df.rename(columns={
        "monomer_name": "Monomer", 'cta_name': 'Chain Transfer Agent', 'res_time': 'Residence Time (min)', 'temperature': 'Temperature (C)', "result": "Conversion"}, inplace=True)
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)
    df = modify_axis_all_visualisations(
        df, 'Residence Time (min)', "round((float(x)/60),3)")
    df = modify_axis_all_visualisations(
        df, 'Conversion', "round(float(x),3)")

    return df


@ csrf_exempt
@ login_required
def all_visualisations(request):
    df = get_all_cleaned_res_time_conversion_data()
    df = clean_data_frame(df)
    x = 'Monomer'
    y = 'Residence Time (min)'
    z = 'Conversion'
    color = "Chain Transfer Agent"
    symbol = "cta_concentration"
    if request.method == "POST":
        x = request.POST.get("choose_x")
        y = request.POST.get("choose_y")
        z = request.POST.get("choose_z")
        x = request.POST.get("choose_x")
        color = request.POST.get("choose_colour")
        symbol = request.POST.get("choose_marker")
        x_input = request.POST.get("x-input")
        y_input = request.POST.get("y-input")
        z_input = request.POST.get("z-input")
        df = modify_axis_all_visualisations(df, x, x_input)
        df = modify_axis_all_visualisations(df, y, y_input)
        df = modify_axis_all_visualisations(df, z, z_input)

    three_d_graph = px.scatter_3d(df, x,
                                  y, z, color, symbol)
    three_d_graph = three_d_graph.to_html()

    column_names = ['cta_concentration', 'monomer_concentration', 'initiator_concentration',
                    'Temperature (C)', 'Residence Time (min)', 'Conversion', 'Chain Transfer Agent', 'Monomer']

    context = {

        'plot_3d_graph': three_d_graph,
        'column_names': column_names

    }
    return render(request, "measurements/all_visualisations.html", context)


def create_descision_tree_from_df(df):
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values.reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=.2, random_state=41)
    regressor = DecisionTreeRegressor(min_samples_split=3)

    kinetics_model = regressor.fit(X_train, Y_train)
    Y_pred = regressor.predict(X_test)
    print(X_train[0])
    print(Y_train[0])

    X_train = np.round(X_train[0].astype(np.float64), 3)
    Y_train = np.round(Y_train[0].astype(np.float64), 3)
    R_square = r2_score(X_train, Y_train)
    print('Coefficient of Determination', R_square)

    error_tree_meansquared = np.sqrt(mean_squared_error(Y_test, Y_pred))
    return [kinetics_model, error_tree_meansquared]


def predict_from_model(fit, input):
    return fit.predict(input)


@ csrf_exempt
@ login_required
def monomer_models(request):

    df_experiments_cta_join = get_all_rate_data()
    predicted_k = "Predicted Rate"
    squared_error = "Mean Square Error"

    df_experiments_cta_join.dropna(subset=['rate'], inplace=True)

    data_target = df_experiments_cta_join[['temperature',	'cta_concentration', 'monomer_Mw', 'monomer_density_g_per_ml', 'monomer_boiling_point_celsius', 'monomer_vapour_pressure_kPa',
                                           'monomer_viscosity_cP', 'monomer_c_number', 'cta_Mw_cta', 'cta_density_g_per_ml_cta', 'cta_reflective_index_cta', 'cta_boiling_point_c_cta', 'cta_c_number_cta', 'rate']]
    model = create_descision_tree_from_df(data_target)[0]
    squared_error = create_descision_tree_from_df(data_target)[1]
    data_target.to_csv("/Users/miladnemati/Desktop/model_features.csv")

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
        prediction_input = [[temp, cx_cm, monomer_mw, monomer_d, monomer_bp, monomer_vp, monomer_v, monomer_nc, cta_mw, cta_d,
                             cta_ri, cta_bp, cta_nc]]
        predicted_k = predict_from_model(model, prediction_input)[0]

    context = {
        "predicted_k": predicted_k,
        "squared_error": squared_error

    }

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


def get_CTA_reaction_data(request, name):
    finale_merged_data = get_all_cleaned_res_time_conversion_data()
    finale_merged_data.replace('', np.nan, inplace=True)
    finale_merged_data.dropna()
    filtered_data_experiments = finale_merged_data.loc[finale_merged_data['monomer_name'] == name]

    return filtered_data_experiments


def get_axis(list_data):

    x = list(list_data['temperature'])
    y = list(list_data['res_time'])
    z = list(list_data['result'])
    CTA = list(list_data['cta_name'])
    cx_ratio = list(list_data['cta_concentration'])

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

        data[axis_label] = new_data
    except:
        print("Not a valid visualisation formula")
        pass

    return data


def get_axis_data(list_data):
    list_data.replace('', np.nan, inplace=True)
    list_data.dropna(inplace=True)
    x, y, z, cx_ratio, CTA = get_axis(list_data)

    data = {
        "temperature(C)": x,
        "residence_time(s)": y,
        "conversion": z,
        "Chain Transfer Agent": CTA,
        "cta_concentration": cx_ratio,

    }
    return data


def plot_2d_kinetic_graph(name):
    all_data = get_all_rate_data()
    data_subset = all_data.loc[all_data['monomer_name'] == name]
    data_subset['temperature'] = pandas.to_numeric(data_subset['temperature'])
    data_subset['rate'] = pandas.to_numeric(data_subset['rate'])

    scatter_plot = px.scatter(data_subset,
                              x='temperature', y='rate', color='cta_name', symbol='cta_concentration', trendline='ols', trendline_scope='overall')
    results = px.get_trendline_results(scatter_plot)
    results = results.iloc[0]["px_fit_results"].summary()
    plot_html_output = scatter_plot.to_html()

    return plot_html_output


def plot_3d_graph(df):

    fig = px.scatter_3d(df, x='temperature(C)',
                        y='residence_time(s)', z='conversion', color="Chain Transfer Agent", symbol="cta_concentration")
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
        left_axis = request.POST.get("choose_x")
        left_input = request.POST.get("x-input")
        middle_axis = request.POST.get("choose_y")
        middle_input = request.POST.get("y-input")
        right_axis = request.POST.get("choose_z")
        right_input = request.POST.get("z-input")

        df = modify_axis_all_visualisations(df, left_axis, left_input)
        df = modify_axis_all_visualisations(df, middle_axis, middle_input)
        df = modify_axis_all_visualisations(df, right_axis, right_input)
    plot_3d = plot_3d_graph(df)
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
    list_data.dropna(inplace=True)

    temperature, y, z, cx_ratio, CTA = get_axis(list_data)
    data = get_axis_data(list_data)

    df = pandas.DataFrame(data)
    df.to_csv("/Users/miladnemati/Desktop/result/list_data.csv")
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
        cx_ratio_chosen = request.POST.get("cta_concentration")
        order_chosen = request.POST.get("order")

        df = df.loc[(df['temperature(C)'] == float(Temperature_chosen)) & (
            df['Chain Transfer Agent'] == CTA_chosen) & (
            df['cta_concentration'] == float(cx_ratio_chosen))]
        try:
            k = determine_rate_of_data_subset(df)
        except:
            print("not valid")
    two_d_graph = plot_2d_kinetic_graph(name)
    three_d_graph = plot_3d_graph(df)
    context = {
        'temperature_list': temperature_list,
        'CTA_list': CTA_list,
        'name': name,
        'reaction_orders': reaction_orders,
        'plot_3d_graph': three_d_graph,
        'plot_2d_graph': two_d_graph,
        'cta_concentration': cx_ratio,
        'k': k
    }

    return render(request, 'measurements/kinetic_view.html', context)
