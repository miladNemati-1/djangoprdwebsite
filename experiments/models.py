from datetime import date
from django.db import models
from users.models import User, Group
from chemicals.models import InChi
from django.utils.html import format_html

class Reactor(models.Model):

    class Type(models.TextChoices):
        BATCH = 'B', 'Batch'
        FLOW = 'F', 'Flow'

    #id             = int, primary key, automatically provided
    name            = models.CharField(max_length=30)
    volume          = models.FloatField(verbose_name="Volume (ml)")
    type            = models.CharField(max_length=1, choices=Type.choices)
    group           = models.ForeignKey(Group, on_delete=models.PROTECT)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Reactor'
        verbose_name_plural = 'Reactors'

class Company(models.Model):
    #id             = int, primary key, automatically provided
    name            = models.CharField(max_length=60)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Company'
        verbose_name_plural = 'Companies'

class Experiment(models.Model):
    #id             = int, primary key, automatically provided
    user            = models.ForeignKey(User, on_delete=models.PROTECT)     #if a user has an experiment the user can't be deleted
    date            = models.DateField(default=date.today)
    time            = models.TimeField()
    name            = models.CharField(max_length=127)                      #your internal identifier for your experiment
    temperature     = models.FloatField(verbose_name="Temperature (ºC)")
    total_volume    = models.FloatField(verbose_name="Volume (ml)")
    reactor         = models.ForeignKey(Reactor, on_delete=models.PROTECT, null=True)  #if a setup has an experiment the setup can't be deleted

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Experiment'
        verbose_name_plural = 'Experiments'

class Inventory(models.Model):
    #id             = int, primary key, automatically provided
    inchi           = models.ForeignKey(InChi, on_delete=models.PROTECT, related_name="getInventory", verbose_name="Name")    #can't delete a chemical that was used as a supply
    company         = models.ForeignKey(Company, on_delete=models.PROTECT)
    purity          = models.FloatField(null=True)
    extra_info      = models.CharField(max_length=511, blank=True)
    url             = models.URLField(max_length=511, blank=True)
    group           = models.ForeignKey(Group, on_delete=models.PROTECT)

    def __str__(self):
        return f"{self.company}: {self.inchi}"

    class Meta:
        verbose_name = 'Inventory'
        verbose_name_plural = 'Inventory'

class Experiment_Chemicals(models.Model):
    
    class Type(models.TextChoices):
        MONOMER = 'M', 'Monomer'
        INITIATOR = 'I', 'Initiator'
        SOLVENT = 'S', 'Solvent'
        OTHER = 'O', 'Other'
    
    experiment      = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    inventory       = models.ForeignKey(Inventory, on_delete=models.CASCADE, related_name="getExperiments")
    type            = models.CharField(max_length=1, choices=Type.choices)
    molarity        = models.FloatField(verbose_name="Molarity (M)")

    def delete_row(self):
        return format_html('<i class="bi bi-trash"></i>')

    def __str__(self):
        return f"{self.inventory}: {self.type}"

    class Meta:
        verbose_name = 'Experiment Chemical'
        verbose_name_plural = 'Experiment Chemicals'