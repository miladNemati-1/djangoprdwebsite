from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include
from users.views import home_page, my_account, research_group, create_group, get_institution_ajax
from chemicals.views import search_chemicals, add_chemical, chemical_details
from experiments.views import add_experiment, my_experiments, my_chemicals, my_equipment, add_equipment, add_supplier, experiment_detail
from measurements.views import monomer_kinetics, upload_file, delete_file, view_3d_graph, view_graph, view_3d_kinetic_graph, monomer_models, all_visualisations

urlpatterns = [
    path('', home_page, name='home'),
    path('admin/', admin.site.urls),
    path('accounts/', include('allauth.urls')),
    path('accounts/me', my_account, name='my_account'),
    path('groups/create', create_group, name='create_group'),
    path('groups/ajax', get_institution_ajax, name='get_institution_ajax'),
    path('groups/<int:pk>', research_group, name='research_group'),
    path('monomers/', search_chemicals, name='search_chemicals'),
    path('monomers/add', add_chemical, name='add_chemical'),
    path('monomers/<int:pk>', chemical_details, name='chemical_details'),
    path('experiments/my_experiments', my_experiments, name="my_experiments"),
    path('experiments/my_experiments/add',
         add_experiment, name="add_experiment"),
    path('experiments/my_experiments/<int:pk>',
         experiment_detail, name="experiment_detail"),
    path('experiments/my_chemicals', my_chemicals, name="my_chemicals"),
    path('measurements/monomer_kinetics',
         monomer_kinetics, name="monomer_kinetics"),
    path('experiments/my_chemicals/add', add_supplier, name="add_supplier"),
    path('experiments/my_equipment', my_equipment, name="my_equipment"),
    path('experiments/my_equipment/add', add_equipment, name="add_equipment"),
    path('measurements/upload/<int:pk>', upload_file, name="upload_file"),
    path('measurements/delete/<int:pk>/<int:path>',
         delete_file, name="delete_file"),
    path('measurements/graph/<int:pk>', view_graph, name='view_graph'),
    path('measurements/view_3d_monomer_graph/<str:name>',
         view_3d_graph, name='view_3d_monomer_graph'),
    path('measurements/view_3d_kinetics_graph/<str:name>',
         view_3d_kinetic_graph, name='view_3d_kinetic_graph'),
    path('measurements/monomer_models/',
         monomer_models, name='monomer_models'),
    path('measurements/all_visualisations/',
         all_visualisations, name='all_visualisations')
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
