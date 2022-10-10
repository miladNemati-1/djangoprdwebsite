from django.contrib import admin

from .models import Measurement, Device

class DeviceAdminConfig(admin.ModelAdmin):

    list_display = ('company', 'model')

admin.site.register(Device, DeviceAdminConfig)
