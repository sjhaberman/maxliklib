//Find variable, form, and position table from array of forms.
#include<armadillo>
using namespace arma;
using namespace std;
struct vn
{
    vector<string>varnames;
    mat vars;
};
struct varforpos
{
    string varname;
    int form;
    int position;
};
struct varlocs
{
    vector<string>catnames;
    vector<int> forms;
    vec offset;
    vector<int> positions;
    vector<string> preds;
    char type;
    char transform;
    mat transition;
    string varname;
};
bool vtabsort(const varforpos & a,const varforpos & b)
   {return(a.varname<b.varname);};
bool vtabchk(const varforpos & a, const varforpos & b)
   {return((a.varname==b.varname)&&(a.form==b.form));};
vector<varlocs> vfp(const vector<vn> & dataf)
{
    int i, j, k, varcounts;
    varforpos u;
    vector<vn>::const_iterator datafit;
    vector<string>::const_iterator varnamesit;
//Maximum possible number of variables.
    varcounts=0;
    for(datafit=dataf.cbegin();datafit!=dataf.cend();++datafit)
    {
        varcounts+=datafit->varnames.size();
    }
    vector<varforpos>vtab(varcounts);
    vector<varforpos>::iterator vtabit,vtabit1,vtabit2;
    vector<varlocs>result(varcounts);
    vector<varlocs>::iterator resultit;
    vector<int>::iterator formit,posit;
    vtabit=vtab.begin();
    i=0;
    for(datafit=dataf.cbegin();datafit!=dataf.cend();++datafit)
    {
        j=0;
        for(varnamesit=datafit->varnames.cbegin();
           varnamesit!=datafit->varnames.cend();
           ++varnamesit)
        {
            vtabit->varname=*varnamesit;
            vtabit->form=i;
            vtabit->position=j;
            ++vtabit;
            j++;
        }
        i++;
    }
    stable_sort(vtab.begin(),vtab.end(),vtabsort);
//Check for variable duplication on form.
    if(distance(adjacent_find(vtab.begin(),vtab.end(),vtabchk),vtab.end())>0)
    {
        cout<<"Duplicate variable name on a form "<<endl;
        result.clear();
        return result;
    }
//Check for where variables in vtab start and end.
    vtabit=vtab.begin();
    for(resultit=result.begin();resultit!=result.end();++resultit)
    {
        u=*vtabit;
        const function <bool(const varforpos & v)>f
           =[&u](const varforpos & v)
        {return vtabsort(u,v);};
        vtabit1=find_if(vtabit,vtab.end(),f);
        i=distance(vtabit,vtabit1);
        resultit->varname=vtabit->varname;
        resultit->forms.resize(i);
        resultit->positions.resize(i);
        formit=resultit->forms.begin();
        posit=resultit->positions.begin();
        for(vtabit2=vtabit;vtabit2!=vtabit1;++vtabit2)
        {
            *formit=vtabit2->form;
            *posit=vtabit2->position;
            ++formit;
            ++posit;
        }
        resultit->type='.';
        resultit->transform='.';
        if(vtabit1==vtab.end())break;
        vtabit=vtabit1;
    }   
//Now for desired table.
    j=distance(result.begin(),resultit);
    result.resize(j+1);
    return result;
}
