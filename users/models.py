from django.db import models
from django.utils import timezone
from django.dispatch import receiver
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager
from allauth.account.signals import email_changed

class UserManager(BaseUserManager):

    def create_user(self, email, first_name, last_name, password, **other_fields):

        if not email:
            raise ValueError('You must provide an email address')

        email = self.normalize_email(email)                 #Make everything lowercase
        user = self.model(email=email, first_name=first_name, last_name=last_name, **other_fields)
        user.set_password(password)
        user.save()
        return user
    
    def create_superuser(self, email, first_name, last_name, password, **other_fields):             #create a new superuser with >>> python manage.py createsuperuser

        other_fields.setdefault('is_staff', True)
        other_fields.setdefault('is_superuser', True)
        other_fields.setdefault('is_active', True)

        return self.create_user(email, first_name, last_name, password, **other_fields)

class User(AbstractBaseUser, PermissionsMixin):
    #id             = int, primary key, automatically provided
    #password       = varchar(128), automatically provided
    #last_login     = datetime, automatically provided
    email           = models.EmailField(max_length=60, unique=True)
    first_name      = models.CharField(verbose_name="First name", max_length=30)
    last_name       = models.CharField(verbose_name="Last name", max_length=30)
    orcid           = models.CharField(verbose_name="ORCID", blank=True, default='', max_length=19)
    date_created    = models.DateTimeField(default=timezone.now)
    is_private      = models.BooleanField(default=True)     #Hides your personal info (name, email, ORCID) for your profile page
    is_active       = models.BooleanField(default=True)     #Designates whether this user should be treated as active. Unselect this instead of deleting accounts. Needs True to send email verification on signup.
    is_staff        = models.BooleanField(default=False)    #Designates whether the user can log into this admin site.
    is_superuser    = models.BooleanField(default=False)    #Designates that this user has all permissions without explicitly assigning them.

    USERNAME_FIELD = 'email'                                #No username is needed, only an email address to log in. See settings.py for more parameters that need to be set.
    REQUIRED_FIELDS = ['first_name', 'last_name']

    objects = UserManager()

    def __str__(self):                                      #custom string reprensentation of the model
        return f"{self.first_name} {self.last_name}"

@receiver(email_changed)        #when the primary email address is changed, update it in our own user database (automatic trigger)
def update_user_email(request, user, from_email_address, to_email_address, **kwargs):
    email = to_email_address

class Group(models.Model):
    #id             = int, primary key, automatically provided
    name            = models.CharField(verbose_name="Research Group", max_length=60)
    short_name      = models.CharField(verbose_name="Short name", max_length=10, blank=True)
    is_private      = models.BooleanField(default=True)     #makes group members visible to everyone vs only own group

    members         = models.ManyToManyField(User, through='User_Group', related_name='res_groups')

    class Meta:
        verbose_name = 'Research Group'
        verbose_name_plural = 'Research Groups'

    def __str__(self):
        return f"{self.name} ({self.short_name})"

class User_Group(models.Model):
    user            = models.ForeignKey(User, on_delete=models.CASCADE)         #if the user gets deleted, also delete the relationship to the group
    group           = models.ForeignKey(Group, on_delete=models.CASCADE)        #if the group gets deleted, remove all the people from that group        
    is_leader       = models.BooleanField(default=False)                        #set flag for group leader(s), who can manage their members

    class Meta:
        verbose_name = 'Group Member'
        verbose_name_plural = 'Group Members'

    def __str__(self):
        if self.is_leader:
            return f"{self.user} - {self.group} - Leader"
        else:
            return f"{self.user} - {self.group}"

class Country(models.Model):                                                    #the country table comes from an online tool, see aux files to populate this table on a new setup
    #id
    code            = models.CharField(max_length=2)
    name            = models.CharField(max_length=60)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Country'
        verbose_name_plural = 'Countries'

class Institution(models.Model):
    #id
    name            = models.CharField(max_length=127)
    short_name      = models.CharField(blank=True, max_length=20)
    country         = models.ForeignKey(Country, on_delete=models.SET_NULL, null=True) #should a country stop existing we don't want to remove the university

    groups          = models.ManyToManyField(Group)

    def __str__(self):
        return f"{self.name} ({self.short_name}), {self.country.name}"

    class Meta:
        verbose_name = 'Institution'
        verbose_name_plural = 'Institutions'