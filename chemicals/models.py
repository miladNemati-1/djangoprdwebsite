from django.db import models
from django.db.models.functions import Length

class InChi(models.Model):
    #id             = int, primary key, automatically provided
    inchi           = models.CharField(verbose_name="InChi", max_length=511)
    inchi_key       = models.CharField(verbose_name="InChiKey", max_length=27, unique=True)
    mw              = models.FloatField(verbose_name="Molecular Weight (g/mol)")
    #img             = models.ImageField

    class Meta:
        verbose_name = 'InChi'
        verbose_name_plural = 'InChi'

    def __str__(self):
        result = self.inchi_name.filter(common_name=True).first()
        short = self.inchi_name.filter(abbreviation=True).first()
        return f"{result.name} ({short.name})" if result and short else result.name if result else self.inchi_key

class CAS(models.Model):
    #id
    inchi           = models.ForeignKey(InChi, on_delete=models.CASCADE, related_name="inchi_cas")
    cas             = models.CharField(max_length=31)

    class Meta:
        verbose_name = 'CAS'
        verbose_name_plural = 'CAS'

    def __str__(self):
        return self.cas

class SMILES(models.Model):
    #id
    inchi           = models.ForeignKey(InChi, on_delete=models.CASCADE)
    smiles          = models.CharField(max_length=511)

    class Meta:
        verbose_name = 'SMILES'
        verbose_name_plural = 'SMILES'

    def __str__(self):
        return self.smiles

class Name(models.Model):
    #id
    inchi           = models.ForeignKey(InChi, on_delete=models.CASCADE, related_name="inchi_name")
    name            = models.CharField(max_length=511)
    iupac           = models.BooleanField(verbose_name="IUPAC", default=False)
    common_name     = models.BooleanField(default=False)
    abbreviation    = models.BooleanField(default=False)

    class Meta:
        verbose_name = 'Name'
        verbose_name_plural = 'Names'

    def __str__(self):
        return self.name