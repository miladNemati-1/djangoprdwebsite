from django.http import JsonResponse
from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

from .forms import CreateGroupForm, CreateInstitutionForm, UpdateUserForm
from .models import Group, Institution, User_Group

def home_page(request):
    return render(request, "home.html")

@login_required
def my_account(request):
    if request.method == 'POST':
        form = UpdateUserForm(request.POST, instance=request.user)

        if form.is_valid():
            form.save()
            messages.success(request, 'Your profile was successfully updated')
            return redirect(to='my_account')
            
    else:
        form = UpdateUserForm(instance=request.user)

    groups = User_Group.objects.filter(user=request.user.id).order_by('-is_leader')
    
    context = {
        'form': form,
        'groups': groups,
    }
    
    return render(request, "users/account.html", context)

@login_required
def research_group(request, pk):

    group = Group.objects.get(id=pk)
    users = User_Group.objects.filter(group=pk)

    context = {
        'group': group,
        'users': users
    }

    return render(request, "users/group.html", context)

@login_required
def create_group(request):
    if request.method == 'POST':
        form_group = CreateGroupForm(request.POST, prefix="grp")                #use prefixes because both forms have "name" and "short_name" fields
        form_institution = CreateInstitutionForm(request.POST, prefix="ins")

        #check if we get a number from the form (id of the institution (str)) or a valid form if the rest of the form is filled correctly (bool)
        form_institution_is_valid = form_institution.data['ins-institution'] or form_institution.is_valid()

        if form_group.is_valid() and form_institution_is_valid:                 #todo: check for group duplication
            g = form_group.save()
            if type(form_institution_is_valid) == bool:                         #if the institution is new and was created through the form
                i = form_institution.save()                                     #save the data from the form and store a reference to the object in the table
            else:                                                               #if the institution was selected via the dropdown menu and we have an id:int
                i = Institution.objects.get(id=form_institution_is_valid)       #get a reference to the object from the table
            u = request.user                                                    #get a reference to the current user, who will be the leader of the new group
            i.groups.add(g)                                                     #add the group to the institution via the ManyToMany relationship
            g.members.add(u, through_defaults={'is_leader': True})              #add the current user as the group leader of the newly created group
            messages.success(request, 'Your group was successfully created')
            return redirect(to='my_account')

    else:
        form_group = CreateGroupForm(prefix="grp")
        form_institution = CreateInstitutionForm(prefix="ins")

    context = {
        'form_group': form_group,
        'form_institution': form_institution
    }

    return render(request, "users/create_group.html", context)

@login_required
def get_institution_ajax(request):                                              #get all data needed to fill the form when picking an institution from the dropdown menu
    
    pk = request.GET.get('pk', None)
    inst = Institution.objects.get(id=pk)

    data = {"name": inst.name, "short_name": inst.short_name, "country": inst.country.id}

    return JsonResponse(data)