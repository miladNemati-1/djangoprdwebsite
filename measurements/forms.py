from django import forms
from .models import Measurement, Device
from experiments.models import Experiment

class AddFileForm(forms.ModelForm):

    def __init__(self, *args, **kwargs):
        self.pk = kwargs.pop("pk")
        super(AddFileForm,self).__init__(*args, **kwargs)
        self.fields["experiment"].queryset = Experiment.objects.all()
        self.fields["experiment"].initial = Experiment.objects.get(id=self.pk)

    device = forms.ModelChoiceField(queryset=Device.objects.all(), empty_label="---------")
    file = forms.FileField()
    is_approved = forms.BooleanField(initial=True)

    class Meta:
        model = Measurement
        fields = ['experiment', 'device', 'file', 'is_approved']