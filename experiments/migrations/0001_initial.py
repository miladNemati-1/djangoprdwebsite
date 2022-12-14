# Generated by Django 4.0.4 on 2022-08-17 01:41

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Company',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=60)),
            ],
            options={
                'verbose_name': 'Company',
                'verbose_name_plural': 'Companies',
            },
        ),
        migrations.CreateModel(
            name='Experiment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField(default=datetime.date.today)),
                ('time', models.TimeField()),
                ('name', models.CharField(max_length=127)),
                ('temperature', models.FloatField(verbose_name='Temperature (ºC)')),
                ('total_volume', models.FloatField(verbose_name='Volume (ml)')),
            ],
            options={
                'verbose_name': 'Experiment',
                'verbose_name_plural': 'Experiments',
            },
        ),
        migrations.CreateModel(
            name='Experiment_Chemicals',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('type', models.CharField(choices=[('M', 'Monomer'), ('I', 'Initiator'), ('S', 'Solvent'), ('O', 'Other')], max_length=1)),
                ('molarity', models.FloatField(verbose_name='Molarity (M)')),
            ],
            options={
                'verbose_name': 'Experiment Chemical',
                'verbose_name_plural': 'Experiment Chemicals',
            },
        ),
        migrations.CreateModel(
            name='Inventory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('purity', models.FloatField()),
                ('extra_info', models.CharField(blank=True, max_length=511)),
                ('url', models.URLField(blank=True, max_length=511)),
            ],
            options={
                'verbose_name': 'Inventory',
                'verbose_name_plural': 'Inventory',
            },
        ),
        migrations.CreateModel(
            name='Reactor',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30)),
                ('volume', models.FloatField(verbose_name='Volume (ml)')),
                ('type', models.CharField(choices=[('B', 'Batch'), ('F', 'Flow')], max_length=1)),
            ],
            options={
                'verbose_name': 'Reactor',
                'verbose_name_plural': 'Reactors',
            },
        ),
    ]
