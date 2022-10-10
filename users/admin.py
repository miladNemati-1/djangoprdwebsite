from django.contrib import admin
from .models import User, Group, Institution, Country, User_Group
from django.contrib.auth.admin import UserAdmin

class UserAdminConfig(UserAdmin):

    search_fields = ('first_name', 'last_name', 'orcid')                                    #list of searchable fields
    list_filter = ()                                                                        #disable filter list
    ordering = ('-id',)                                                                     #order by id, descending order
    list_display = ('first_name', 'last_name', 'orcid', 'is_staff', 'is_active')            #only show these values in the table

    fieldsets = (
        (None, {'fields': ('email', 'first_name', 'last_name', 'orcid')}),
        ('Permissions', {'fields': ('is_active', 'is_staff')})
    )

    add_fieldsets = (                                               #list of data you want to add to the database when making a new user.
        (None, {
            'fields': ('email', 'first_name', 'last_name', 'orcid', 'password1', 'password2', 'is_active', 'is_staff')}
         ),
    )

admin.site.register(User, UserAdminConfig)

class GroupAdminConfig(admin.ModelAdmin):

    list_display = ('name', 'short_name', 'is_private')
    ordering = ('name',)

admin.site.register(Group, GroupAdminConfig)

class InstitutionAdminConfig(admin.ModelAdmin):

    list_display = ('name', 'short_name', 'country')
    ordering = ('country',)

admin.site.register(Institution, InstitutionAdminConfig)

class CountryAdminConfig(admin.ModelAdmin):

    list_display = ('code', 'name')
    ordering = ('code',)

admin.site.register(Country, CountryAdminConfig)

class UserGroupAdminConfig(admin.ModelAdmin):

    list_display = ('user', 'group', 'is_leader')
    ordering = ('group', '-is_leader', 'user')

admin.site.register(User_Group, UserGroupAdminConfig)