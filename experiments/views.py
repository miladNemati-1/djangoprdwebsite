from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

from experiments.forms import AddEquipmentForm, AddChemicalForm, AddExperimentForm, AddIngredientForm
from measurements.models import Measurement
from .models import Experiment, Experiment_Chemicals, Reactor, Inventory
from .tables import EquipmentTable, ExperimentTable, SupplierTable, SuppliesTable
from measurements.forms import AddFileForm
from measurements.tables import MeasurementTable


@login_required
def my_experiments(request):

    experiments = ExperimentTable(Experiment.objects.all())

    context = {
        'experiments': experiments,
    }

    return render(request, "experiments/my_experiments.html", context)


@login_required
def my_chemicals(request):

    chemicals = SupplierTable(Inventory.objects.all())

    context = {
        'chemicals': chemicals,
    }

    return render(request, "experiments/my_chemicals.html", context)


@login_required
def my_equipment(request):

    equipment = EquipmentTable(Reactor.objects.all())

    context = {
        'equipment': equipment,
    }

    return render(request, "experiments/my_equipment.html", context)


@login_required
def add_equipment(request):
    if request.method == 'POST':
        form = AddEquipmentForm(request.POST, request=request)

        if form.is_valid():
            form.save()
            messages.success(request, 'Your equipment was successfully added')
            return redirect(to='my_equipment')

    else:
        form = AddEquipmentForm(request=request)

    context = {
        'form': form
    }

    return render(request, "experiments/add_equipment.html", context)


@login_required
def add_supplier(request):
    if request.method == 'POST':
        form = AddChemicalForm(request.POST, request=request)

        if form.is_valid():
            form.save()
            messages.success(request, 'Your chemical was successfully added')
            return redirect(to='my_chemicals')

    else:
        form = AddChemicalForm(request=request)

    context = {
        'form': form
    }

    return render(request, "experiments/add_chemical.html", context)


@login_required
def add_experiment(request):
    if request.method == 'POST':
        form = AddExperimentForm(request.POST)

        if form.is_valid():
            form.save()
            messages.success(request, 'Your experiment was successfully added')
            return redirect(to='my_experiments')
        else:
            print(form.errors)

    else:
        form = AddExperimentForm()

    context = {
        'form': form
    }

    return render(request, "experiments/add_experiment.html", context)


@login_required
def experiment_detail(request, pk):
    if request.method == 'POST':
        form = AddIngredientForm(request.POST, pk=pk)

        if form.is_valid():
            form.save()
            return redirect('experiment_detail', pk=pk)

    else:
        form = AddIngredientForm(pk=pk)

    # make sure this is not iterable: set correct permissions!
    experiment = Experiment.objects.get(id=pk)
    supplies = SuppliesTable(
        Experiment_Chemicals.objects.filter(experiment=pk))
    file_form = AddFileForm(pk=pk)
    files_list = MeasurementTable(Measurement.objects.filter(experiment=pk))

    context = {
        'experiment': experiment,
        'files_list': files_list,
        'file_form': file_form,
        'supplies': supplies,
        'form': form,
        'pk': pk,
    }

    return render(request, "experiments/experiment.html", context)


@login_required
def experiment_detail_delete(request, pk):
    if request.method == 'POST':
        return redirect('experiment_detail', pk=pk)
