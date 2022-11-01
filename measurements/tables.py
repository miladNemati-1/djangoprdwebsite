from cgitb import html
from django_tables2 import tables, TemplateColumn
from .models import Measurement, Monomer


class MeasurementTable(tables.Table):

    class Meta:
        model = Measurement
        attrs = {'class': 'table', 'td': {'class': 'align-middle'}}
        fields = ['file', 'device', 'delete']

    delete = TemplateColumn(template_name='measurements/delete_file_button.html',
                            verbose_name='', attrs={'td': {'align': 'right'}})


class MonomerTable(tables.Table):
    class Meta:
        model = Monomer
        fields = ("name", "Mw")
        attrs = {"class": "table table-hover"}
        # row_attrs = {
        #     "onClick": lambda record: "document.location.href='/measurements/view_3d_monomer_graph/{0}';".format(record.name)}
    delete = TemplateColumn(template_name='measurements/view3d.html',
                            verbose_name='3D VIEW')
    kinetics = TemplateColumn(template_name='measurements/view_kinetic_values_graph.html',
                              verbose_name='Kinetic VIEW')
