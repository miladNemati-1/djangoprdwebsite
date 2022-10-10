from django_tables2 import tables, TemplateColumn
from .models import Measurement

class MeasurementTable(tables.Table):

    class Meta:
        model = Measurement
        attrs = {'class': 'table', 'td': {'class': 'align-middle'}}
        fields = ['file', 'device', 'delete']

    delete = TemplateColumn(template_name='measurements/delete_file_button.html', verbose_name='', attrs={'td': {'align': 'right'}})