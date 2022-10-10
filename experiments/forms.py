from django import forms
from experiments.models import Experiment, Experiment_Chemicals, Reactor, Inventory, Company
from users.models import User, User_Group, Group
from chemicals.models import InChi

class AddEquipmentForm(forms.ModelForm):
    #we need to pass the request to get the user so we can get all the groups that the user is part of
    #we search the User_Group table for a list of group ids via .values_list('group') and use that list as a filter for Group ids
    def __init__(self, *args, **kwargs):
        self.request = kwargs.pop("request")
        super(AddEquipmentForm,self).__init__(*args, **kwargs)
        self.fields["group"].queryset = Group.objects.filter(id__in=User_Group.objects.filter(user=self.request.user).values_list('group'))

    TYPE_CHOICES = (
        ('-', '---------'),
        ('B', 'Batch'),
        ('F', 'Flow'),
    )

    name = forms.CharField()
    volume = forms.FloatField()
    type = forms.ChoiceField(choices = TYPE_CHOICES)

    class Meta:
        model = Reactor
        fields = ['name', 'volume', 'type', 'group']

class AddChemicalForm(forms.ModelForm):
    #we need to pass the request to get the user so we can get all the groups that the user is part of
    #we search the User_Group table for a list of group ids via .values_list('group') and use that list as a filter for Group ids
    def __init__(self, *args, **kwargs):
        self.request = kwargs.pop("request")
        super(AddChemicalForm,self).__init__(*args, **kwargs)
        self.fields["group"].queryset = Group.objects.filter(id__in=User_Group.objects.filter(user=self.request.user).values_list('group'))

    inchi = forms.ModelChoiceField(queryset=InChi.objects.all(), empty_label="---------")
    company = forms.ModelChoiceField(queryset=Company.objects.order_by('name'), empty_label="---------")
    purity = forms.FloatField(required=False)
    extra_info = forms.CharField(required=False, max_length=511, widget=forms.Textarea(attrs={"rows":5}))
    url = forms.URLField(required=False, max_length=511)

    class Meta:
        model = Inventory
        fields = ['inchi', 'company', 'purity', 'extra_info', 'url', 'group']

class AddExperimentForm(forms.ModelForm):

    user = forms.ModelChoiceField(queryset=User.objects.all(), empty_label="---------")     #TODO only yourself as user/ other group members as well?
    date = forms.DateField()
    time = forms.TimeField()
    name = forms.CharField(max_length=127)
    temperature = forms.FloatField()
    total_volume = forms.FloatField()
    reactor = forms.ModelChoiceField(queryset=Reactor.objects.all(), empty_label="---------", required=False)   #TODO only equipment from the group is allowed

    class Meta:
        model = Experiment
        fields = ['user', 'date', 'time', 'name', 'temperature', 'total_volume', 'reactor']

class AddIngredientForm(forms.ModelForm):

    def __init__(self, *args, **kwargs):
        self.pk = kwargs.pop("pk")
        super(AddIngredientForm,self).__init__(*args, **kwargs)
        self.fields["experiment"].queryset = Experiment.objects.all()
        self.fields["experiment"].initial = Experiment.objects.get(id=self.pk)

    TYPE_CHOICES = (
        ('-', '---------'),
        ('M', 'Monomer'),
        ('I', 'Initiator'),
        ('S', 'Solvent'),
        ('O', 'Other'),
    )

    inventory = forms.ModelChoiceField(queryset=Inventory.objects.all(), empty_label="---------")
    type = forms.ChoiceField(choices=TYPE_CHOICES)
    molarity = forms.FloatField()

    class Meta:
        model = Experiment_Chemicals
        fields = ['experiment', 'inventory', 'type', 'molarity']