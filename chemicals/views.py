from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from itertools import chain
from django.db.models.functions import Length
from django.contrib import messages

from .forms import SearchForm
from .models import InChi, CAS, SMILES, Name
from experiments.models import Experiment
from measurements.models import Measurement, Data
from fuzzywuzzy import fuzz
import pandas
from plotly.offline import plot
import plotly.graph_objects as go

@login_required
def search_chemicals(request):
    if request.method == 'POST':
        query = request.POST.get('query')

        inchi_search = InChi.objects.filter(Q(inchi__icontains=query) | Q(inchi_key__icontains=query))
        cas_search = CAS.objects.filter(cas__icontains=query)
        smiles_search = SMILES.objects.filter(smiles__icontains=query)
        name_search = Name.objects.filter(name__icontains=query)

        search = chain(inchi_search, cas_search, smiles_search, name_search)
        
        results = []
        data = {"result": []}

        item_found = False

        for item in search:
            item_found = True
            if (type(item).__name__ == "InChi"):
                results.append([item.id, fuzz.ratio(query,str(item))])          #get the ID from the main table directly
            else:
                results.append([item.inchi.id, fuzz.ratio(query,str(item))])    #get the ID via the foreign key in the additional table

        if not item_found:
            messages.warning(request, 'No matches were found')                  #doesn't trigger?
            return redirect(to='search_chemicals')

        df = pandas.DataFrame(results)
        df.columns = ['id','score']
        df = df.sort_values(by='score', ascending=False).drop_duplicates('id').head(5).reset_index(drop=True)

        for index, row in df.iterrows():

            inchi = InChi.objects.get(id=row['id'])
            cas = CAS.objects.filter(inchi=inchi.id).annotate(l=Length('cas')).order_by('l').first()        #get the shortest CAS since this is probably the 'official' one
            smiles = SMILES.objects.filter(inchi=inchi.id).first()                                          #get a smiles notation, doesn't matter which one
            common_name = Name.objects.filter(inchi=inchi.id, common_name=True).first()
            abbreviation = Name.objects.filter(inchi=inchi.id, abbreviation=True).first()

            info = {"pk": inchi.id,
                    "inchi": inchi.inchi,
                    "inchi_key": inchi.inchi_key,
                    "cas": '' if cas is None else cas.cas,
                    "smiles": '' if smiles is None else smiles.smiles,
                    "common_name": '' if common_name is None else common_name.name,
                    "abbreviation": '' if abbreviation is None else '(' + abbreviation.name + ')',
                    "score": str(row['score']),
            }

            data["result"].append(info)

        return JsonResponse(data)

    context = {

    }

    return render(request, "chemicals/search.html", context)

@login_required
def add_chemical(request):

    form = SearchForm(request.POST or None)

    if request.method == 'POST' and form.is_valid():

        inchi_obj = InChi(
            inchi = form.cleaned_data['inchi'],
            inchi_key = form.cleaned_data['inchi_key'],
            mw = form.cleaned_data['mw']
        )
        inchi_obj.save()

        #save IUPAC name (if it exists)
        if form.cleaned_data['iupac']:
            to_save = Name(
                inchi = inchi_obj,
                name = form.cleaned_data['iupac'],
                iupac = True
            )
            to_save.save()

        #save common name (if it exists)
        if form.cleaned_data['common_name']:
            to_save = Name(
                inchi = inchi_obj,
                name = form.cleaned_data['common_name'],
                common_name = True
            )
            to_save.save()

        #save abbreviation (if it exists)
        if form.cleaned_data['abbreviation']:
            to_save = Name(
                inchi = inchi_obj,
                name = form.cleaned_data['abbreviation'],
                abbreviation = True
            )
            to_save.save()
    
        #save other names (if they exist)
        if form.cleaned_data['other_names']:
            for other_name in form.cleaned_data['other_names']:
                to_save = Name(
                    inchi = inchi_obj,
                    name = other_name,
                )
                to_save.save()

        #save CAS number (if they exist)
        if form.cleaned_data['cas']:
            for cas in form.cleaned_data['cas']:
                to_save = CAS(
                    inchi = inchi_obj,
                    cas = cas,
                )
                to_save.save()

        #save SMILES (if they exist)
        if form.cleaned_data['smiles']:
            for smiles in form.cleaned_data['smiles']:
                to_save = SMILES(
                    inchi = inchi_obj,
                    smiles = smiles,
                )
                to_save.save()

        messages.success(request, 'Your chemical was added to the database')
        return redirect(to='search_chemicals')

    context = {
        'form': form,
    }

    return render(request, "chemicals/add.html", context)

@login_required
def chemical_details(request, pk):

    test = InChi.objects.get(id=pk).getInventory.all()
    name = InChi.objects.get(id=pk)

    print(name)

    y = Experiment.objects.none()
    for t in test:
        y = y | t.getExperiments.all()

    z = Experiment.objects.none()
    for x in y:
        z = z | Experiment.objects.filter(id=x.experiment.id)

    a = Measurement.objects.none()
    for b in z:
        a = a | b.getMeasurement.all()

    d = Data.objects.none()
    for c in a:
        d = d | c.getData.all()

    print(d.count())

    df = pandas.DataFrame(list(d.values()))

    graph_conv = go.Scatter(x=df.res_time, y=df.result, mode='markers')

    #pad for centering and round to nearest 100
    max = df['res_time'].max() + df['res_time'].min()
    max -= max % -100             

    layout_conv = {
        'title': str(name) + ": conversion",
        'xaxis_title': 't_res (s)',
        'xaxis_range': [0, max],        
        'yaxis_title': 'conversion',
        'yaxis_range': [0, 0.6],
        'height': 630,
        'width': 840,
        'paper_bgcolor': 'rgba(0,0,0,0)',
    }

    plot_conv = plot({'data': graph_conv, 'layout': layout_conv}, output_type='div')

    context = {
        'plot_conv': plot_conv,
    }

    return render(request, "chemicals/details.html", context)