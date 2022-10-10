from django import forms
from allauth.account.forms import SignupForm
from users.models import Group, Institution, User, Country
from django.core.validators import RegexValidator

class CustomSignupForm(SignupForm):

    first_name = forms.CharField(min_length=2, max_length=30, required=True, widget=forms.TextInput(attrs={'placeholder': 'Marie'}))
    last_name = forms.CharField(min_length=2, max_length=30, required=True, widget=forms.TextInput(attrs={'placeholder': 'Curie'}))
    email = forms.EmailField(max_length=60, required=True, widget=forms.EmailInput(attrs={'placeholder': 'marie.curie@hotmail.com'}))

    def save(self, request):

        user = super(CustomSignupForm, self).save(request)
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.save()
        return user

class UpdateUserForm(forms.ModelForm):

    first_name = forms.CharField(min_length=2, max_length=30, required=True)
    last_name = forms.CharField(min_length=2, max_length=30, required=True)
    email = forms.EmailField(max_length=60, required=True)
    orcid = forms.CharField(label="ORCID", max_length=19, required=False, 
        validators=[RegexValidator('^\d{4}-\d{4}-\d{4}-(\d{3}X|\d{4})$', message="This is not a valid ORCID. See orcid.org for more info.")], 
        widget=forms.TextInput(attrs={'placeholder': '0000-0000-0000-0000'}))

    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email', 'orcid']

class CreateGroupForm(forms.ModelForm):

    name = forms.CharField(label="Group name", max_length=60, required=True)
    short_name = forms.CharField(max_length=10, required=False)
    is_private = forms.BooleanField(label="Private group", initial=True)

    class Meta:
        model = Group
        fields = ['name', 'short_name', 'is_private']

class CreateInstitutionForm(forms.ModelForm):

    institution = forms.ModelChoiceField(queryset=Institution.objects.all().order_by('name'), required=False)

    name = forms.CharField(max_length=127, required=True)
    short_name = forms.CharField(max_length=20, required=False)
    country = forms.ModelChoiceField(queryset=Country.objects.all(), empty_label="")

    class Meta:
        model = Institution
        fields = ['name', 'short_name', 'country']