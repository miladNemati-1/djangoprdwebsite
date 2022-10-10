import django_tables2 as tables
from .models import Experiment, Experiment_Chemicals, Reactor, Inventory

class ExperimentTable(tables.Table):
    class Meta:
        model = Experiment
        fields = ("user", "name", "temperature", "total_volume", "reactor", )
        attrs = {"class": "table table-hover"}
        row_attrs = {"onClick": lambda record: "document.location.href='/experiments/my_experiments/{0}';".format(record.id)}       #URL is not dynamic here

class SupplierTable(tables.Table):
    class Meta:
        model = Inventory
        fields = ("inchi", "company", "purity", )
        #attrs = {"class": "table table-hover"}    #disabled while not clickable

class SuppliesTable(tables.Table):

    delete_button = tables.Column(accessor="delete_row", verbose_name="")

    class Meta:
        model = Experiment_Chemicals
        fields = ("inventory", "type", "molarity")

class EquipmentTable(tables.Table):
    class Meta:
        model = Reactor
        fields = ("name", "volume", "type")
        #attrs = {"class": "table table-hover"}    #disabled while not clickable