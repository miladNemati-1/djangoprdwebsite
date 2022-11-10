from django.db import models
from requests import delete
from experiments.models import Experiment
from django.utils.html import format_html


class Device(models.Model):
    # id
    company = models.CharField(max_length=127)
    model = models.CharField(max_length=127)

    def __str__(self):
        return f"{self.company}: {self.model}"

    class Meta:
        verbose_name = 'Device'
        verbose_name_plural = 'Devices'


class Measurement(models.Model):
    # id
    experiment = models.ForeignKey(
        Experiment, on_delete=models.PROTECT, related_name="getMeasurement")
    device = models.ForeignKey(Device, on_delete=models.PROTECT)
    file = models.FileField(upload_to='measurements/')
    is_approved = models.BooleanField(
        default=True)  # set to False in production
    # type (NMR, Mn, ...) -> is this a Device or Measurement attribute?

    def __str__(self):
        return self.file.name

    class Meta:
        verbose_name = 'Measurement'
        verbose_name_plural = 'Measurements'


class Data(models.Model):
    # id

    # if a measurement gets deleted, delete the data as well.
    measurement = models.ForeignKey(
        Measurement, on_delete=models.CASCADE, related_name="getData")
    res_time = models.FloatField()
    result = models.FloatField()
    # is_outlier     a function can identify outliers and set this flag so this data can be hidden if needed


class Monomer(models.Model):
    name = models.CharField(verbose_name="Monomer", max_length=511)
    # if a setup has an experiment the setup can't be deleted
    Mw = models.CharField(max_length=511)
    density_g_per_ml = models.CharField(max_length=511)
    boiling_point_celsius = models.CharField(max_length=511)
    vapour_pressure_kPa = models.CharField(max_length=511)
    viscosity_cP = models.CharField(max_length=511)
    c_number = models.CharField(max_length=511)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Monomer'
        verbose_name_plural = 'Monomers'


class cta(models.Model):
    name = models.CharField(verbose_name="Monomer", max_length=511)
    # if a setup has an experiment the setup can't be deleted
    Mw_cta = models.CharField(max_length=511)
    density_g_per_ml_cta = models.CharField(max_length=511)
    reflective_index_cta = models.CharField(max_length=511)
    boiling_point_c_cta = models.CharField(max_length=511)
    c_number_cta = models.CharField(max_length=511)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'cta'
        verbose_name_plural = 'cta'
