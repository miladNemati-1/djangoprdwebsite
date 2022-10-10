from django.contrib import admin

from .models import InChi, CAS, SMILES, Name

class InChiAdminConfig(admin.ModelAdmin):

    list_display = ('id', 'inchi_key', 'mw', 'inchi')

admin.site.register(InChi, InChiAdminConfig)

class CASAdminConfig(admin.ModelAdmin):

    list_display = ('inchi', 'cas')

admin.site.register(CAS, CASAdminConfig)

class SMILESAdminConfig(admin.ModelAdmin):

    list_display = ('inchi', 'smiles')

admin.site.register(SMILES, SMILESAdminConfig)

class NameAdminConfig(admin.ModelAdmin):

    list_display = ('inchi', 'name', 'common_name', 'iupac', 'abbreviation')
    ordering = ('inchi', '-common_name', '-iupac', '-abbreviation')

admin.site.register(Name, NameAdminConfig)