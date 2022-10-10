from django import forms

class CustomMultipleChoiceField(forms.MultipleChoiceField):
    def clean(self, value):
        return value

class SearchForm(forms.Form):

    iupac = forms.CharField(required=False)
    common_name = forms.CharField(required=False)
    abbreviation = forms.CharField(required=False, max_length=10)
    other_names = CustomMultipleChoiceField(required=False)       #override MCF: options should load on form intialisation

    inchi = forms.CharField()
    inchi_key = forms.CharField()
    mw = forms.CharField()          #should be FloatField but then arrows show, check if float during validation?/ hide arrows with css

    cas = CustomMultipleChoiceField(required=False)
    smiles = CustomMultipleChoiceField(required=False)