//General model for maximum likelihood for latent structures.
//Multiple forms are possible, and missing data
//may exist.  All data variables are read as double-precision numbers. If a variable entry is blank or not
//numeric, then it has value NaN.
//The model and data specifications are provided by the control file.
//The name of the file is read from standard input. 
//The file includes multiple lines of text, with each line a control statement.
//Each statement includes two strings separated by
//a blank character. Each of these strings contains no blank character.  The first string specifies
//the type of control statement, and the second string includes specifications.
//These specifications consist of one or more segments.  If more than one segment is
//present, then segments are separated by semicolons.  Each segment is defined by use of a keyword followed
//by an equal sign and a value corresponding to the keyword.
//The following types of control statements are used.
//
//form, latent, vargroup, variable, cluster, model, parameters,
//quadrature, algorithm, startfile, outfile.
//
//Statements are defined as follows:
//
//form
//    There are one or more form statements.  The statement specifies a data file and the file that
//    provides variable names.  Data files are input files for double-precision matrices
//    that are defined as in the Armadillo documentation.  Each matrix row corresponds to an observation,
//    and each matrix column corresponds to a data variable.  The keyword
//    data specifies the name of the data file.  For example, one might have data=form1.csv.  If no other keyword
//    is present, then the data file is a comma-separated file with a header that provides a variable name for
//    each column.  In other cases, a second keyword variables specifies the name of the variable file that provides
//    names of variables in the data file.  The variable file is a text file.  It can consist of a single
//    space-delimited record with a variable name for each column of the data file or can contain one record
//    for each column of the data file.  In both cases, names cannot contain spaces.
//    For example, one might have variables=names.txt.
//
//latent
//    There is at most one latent statement.  This statement provides the names of the latent variable
//    in comma-separated form.
//vargroup
//    A vargroup statement provides a template for one or more variables used in analysis.
//    The statement provides elements of a varlocs struct that are common to these variables.
//    Common elements may include catnames, preds, transform, and type.
//    Definitions of transform and type are as in genresp.cpp,
//    except that type='W' corresponds to a weight variable and type='I' corresponds to a numerical identifier for an
//    observation.  catnames are value names for a polytomous variable in comma-separated format.
//    preds are predicting variables
//    in comma-separated format.  In addition, vargroupname is the name for the group.
//sf startfile
//   startfile is the file containing the vector of starting values.  If this
//   file is not given, then the default procedure is used.
//
//outputfile outfile
//   outfile is the name of the output file.  If this file is not specified,
//   then no output file is produced.
//
//pflag value
//   value is is false if nothing is printed in ascii form.  Otherwise printing in ascii
//   form is used.  If the pair is omitted, printing occurs.
//
//method algorithm
//   algorithm is G for gradient ascent, C for conjugate gradient ascent,
//   N for modified Newton-Raphson, and L for Louis approximation.  The default is N.
//
//tol value
//   value is the convergence criterion.  It is 0.001 if the pair is omitted.
//
//adapt aq
//   aq is true for adaptive quadrature and false otherwise.  If the pair is omitted,
//   then adaptive quadrature is used.
//
//quad qu
//   qu is G for Gauss-Hermite quadrature and Q for normal quantiles.
//   The default is G.
//  
//points n
//   n is the number of quadrature points per dimension.  The default values are
//   9 for one dimension, 7 for two dimensions, 5 for three dimensions, and 3 otherwise.


//dimlat dimension, where dimension is the dimension of the normal latent variable.  The
//default is 1.  If dimension exceeds 1, then a pair 
//dimnames dimfile
//is used, where dimfile is a file that gives the names of the dimensions.
//For each dimension, a pair
//dimresp variablefile
//is used with variablefile is a file that lists the response variables for the dimension.
//slopes constant, where constant is true if slopes are constant for each dimension and false
//otherwise.  The default is false.
#include<armadillo>
#include<string>
#include<set>
using namespace std;
using namespace arma;
struct clust
{
    string clustername;
    vector<string>varnames;
};
struct vn
{
    vector<string>varnames;
    mat vars;
};
struct model 
{
    char type;
    char transform;
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
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct resp
{
    ivec iresp;
    vec dresp;
};
//Select elements of vector.  all indicates all elements.  list lists elements.
struct xsel
{
    bool all;
    uvec list;
};
//Select elements of matrix.  all indicates all elements.  list lists elements in columns.
struct xselv
{
    bool all;
    umat list;
};
//Specify a model.
//choice is model distribution.
//o is constant vector.
//x is tranformation from beta elements used to lambda that does not involve theta.
//c is transformation from beta elements used and theta double elements  used to lambda.
struct pattern
{
    model choice;
    vec o;
    mat x;
    cube c;
}; 
//Basic quadrature rule.
struct pw
{
    vec points;
    vec weights;
};
// Weights and points for prior.
struct pwr
{
    double weight;
    double kernel;
    resp theta;
};
struct vecmat
{
    vec v;
    mat m;
};
// Adaptive quadrature specifications.
// The choice to use is indicated by adapt, linselect shows the elements involved.
// quadselect shows the quadratic elements involved.
struct adq
{
    bool adapt;
    xsel linselect;
    xselv quadselect;
};
//Adaptive quadrature transformation.
struct dovecmat
{
    double s;
    vec v;
    mat m;
};
//Parameters for function maximization.
struct params
{
    bool print;
    int maxit;
    int maxits;
    double eta;
    double gamma1;
    double gamma2;
    double kappa;
    double tol;
};
struct maxf2v
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
field<string> getcontrolfile(const string & );
vector<vn>getfiles(const vector<vector<string>> & );
vector<vector<string>> keysort(const field<string> & );
vector<vector<string>> keyvar(const string & , const vector<vector<string>> & );
string keylookup(const string & , const vector<vector<string>> & );
vector<varlocs>::iterator varlookup(const string & , vector<varlocs> & );
vector<varlocs> vfp(const vector<vn> & );
vector<string>parse(const string & , const char & );
bool dupname(const vector<string> & );
pair<vector<varlocs>::iterator,vector<varlocs>::iterator> typelookup(const char & ,  vector<varlocs> & );
vector<vector<int>>obspat(const vector<vn> & ,const vector<varlocs> & , const vector<int> & );
bool varfinite(const string & , vector<varlocs> & , vector<vector<int>> & );
void savmaxf2v(const int & , const maxf2v & , const string & , const bool & , const bool & );
pw hermpw(const int & );
pw qnormpwe(const int & );
vecmat genprods(const imat & , const field<pw> & );
imat genfact(const ivec & );
int main()
{
//Result structure.
    maxf2v results;
//Begin by finding control file location in standard input.
    string controlfile;
    try{cin>>controlfile;}
    catch(...){cout<<"Name of control file not read."<<endl; return 1;}
//Get control file.
    field<string>control;
    control=getcontrolfile(controlfile);
    if(control.empty())
    {
        cout<<"Empty control file."<<endl;
        return 1;
    }
    if(control.n_cols!=2)
    {
        cout<<"Control file entries must have two strings."<<endl;
        return 1;
    }
//Sort control file by lexicographical order of keys.
    bool flag;
    int h, i, j, k, m, nc;
    nc=control.n_rows;
    vector<vector<string>>controlvec(nc);
    controlvec=keysort(control);
    if(controlvec.empty())return 1;
//Check for duplicates.
    if(nc>1)
    {
        if(distance(adjacent_find(controlvec.begin(),controlvec.end()),controlvec.end())>0)
        {
            cout<<"Duplicate control file entry."<<endl;
            return 1;
        }
    }
//Find form files in control file.  At least one is needed.
    vector<vector<string>>datafiles;
    vector<vector<string>>::iterator datafilesit;
    vector<string>::iterator datafilesit1;
    string form="form";
    datafiles=keyvar(form,controlvec);
    if(datafiles.empty())
    {
        cout<<"Invalid form specifications."<<endl;
        return 1;
    }
    for(datafilesit=datafiles.begin();datafilesit!=datafiles.end();++datafilesit)
    {
        if(datafilesit->size()>4)
        {
            cout<<"Invalid form specifications."<<endl;
            return 1;
        }
        datafilesit1=datafilesit->begin();
        if(*datafilesit1!="data")
        {
            cout<<"Invalid form specifications."<<endl;
            return 1;
        }
        if(datafilesit->size()==4)
        {
            datafilesit1=2+datafilesit1;
            if(*datafilesit1!="variables")
            {
                cout<<"Invalid form specifications."<<endl;
                return 1;
            }
        }
    }
//Get observed data.
    vector<vn> dataf(datafiles.size());
    dataf=getfiles(datafiles);
//Need data.
    if(dataf.empty()) return 1;
//Get variable, form, position table, and variable map.
    vector<varlocs>vartab;
    vector<varlocs>::iterator vartabit;
    vartab=vfp(dataf);
    if(vartab.empty())return 1;
    vector<string>varnames(vartab.size());
    vector<string>::iterator varnamesit;
    vartabit=vartab.begin();
    for(varnamesit=varnames.begin();varnamesit!=varnames.end();++varnamesit){*varnamesit=vartabit->varname;++vartabit;}
//Get number of observations and form placement.
    int nobs=0, numforms;
    numforms=dataf.size();
    ivec cumobs(numforms),formcount(numforms);
    ivec::iterator formcountit;
    vector<vn>::iterator datafit;
    datafit=dataf.begin();
    for(formcountit=formcount.begin();formcountit!=formcount.end();++formcountit)
    {
        *formcountit=datafit->vars.n_rows;
        ++datafit;
    }
    cumobs=cumsum(formcount);
    nobs=cumobs(numforms-1);
//Get any latent variables.
    vector<varlocs>lattab;
    vector<varlocs>::iterator lattabit;
    vector<string>latnames;
    vector<string>::iterator latnamesit;
    vector<vector<string>>latfile;
    vector<vector<string>>::iterator latfileit;
    vector<string>::iterator latfileit1;
    string latent="latent";
    latfile=keyvar(latent,controlvec);
    if(latfile.size()>1)
    {
        cout<<"No more than one latent file allowed."<<endl;
        return 1;
    }
    if(latfile.size()>0)
    {
        latfileit=latfile.begin();
        latfileit1=latfileit->begin();
        if(*latfileit1!="variables")
        {
            cout<<"Invalid specification of latent variable names"<<endl;
            return 1;
        }
        ++latfileit1;
        latnames=parse(*latfileit1,',');
        sort(latnames.begin(),latnames.end());
        if(distance(adjacent_find(latnames.begin(),latnames.end()),latnames.end())>0)
        {
            cout<<"Duplicate latent variable."<<endl;
            return 1;
        }
//Check that latent variables are not observed variables.
        for(latnamesit=latnames.begin();latnamesit!=latnames.end();++latnamesit)
        {
            if(binary_search(varnames.begin(),varnames.end(),*latnamesit))
            {
                cout<<"Latent variable "<<*latnamesit<<" is also an observed variable."<<endl;
                return 1;
            }
        }
//Set up latent variables in lattab.
        lattab.resize(latnames.size());
        lattabit=lattab.begin();
        for(latnamesit=latnames.begin();latnamesit!=latnames.end();++latnamesit){lattabit->varname=*latnamesit;++lattabit;}
    }
//Set up variable groups.
    vector<vector<string>>vargroupf;
    vector<vector<string>>::iterator vargroupfit;
    string vargroup="vargroup";
    vargroupf=keyvar(vargroup,controlvec);
//The elements of vargroupf are rearranged so that vargroupname comes first.  Sorting is then performed.
    for(vargroupfit=vargroupf.begin();vargroupfit!=vargroupf.end();++vargroupfit)
    {
        rotate(vargroupfit->begin(),next(vargroupfit->end(),-2),vargroupfit->end());
    }
    sort(vargroupf.begin(),vargroupf.end());
    vector<varlocs> vargroupvn(vargroupf.size());
    vector<varlocs>::iterator vargroupvnit;
    vector<string>::iterator vgroupit;
    vector<string>predname;
    vector<string>::iterator prednameit;
    vector<char>transforms={'G','H','L','N'};
    vector<char>types={'B','C','D','E','G','H','I','L','M','N','P','R','S','T','W'};
    vargroupvnit=vargroupvn.begin();
    for(vargroupfit=vargroupf.begin();vargroupfit!=vargroupf.end();++vargroupfit)
    {
        if(vargroupfit!=vargroupf.begin())++vargroupvnit;
        vgroupit=vargroupfit->begin();
        ++vgroupit;
        vargroupvnit->varname=*vgroupit;
        ++vgroupit;
        if(*vgroupit=="catnames")
        {
            vgroupit=++vgroupit;
            vargroupvnit->catnames=parse(*vgroupit,',');
//Check for distinct categories.
            if(vargroupvnit->catnames.size()<2)
            {
                cout<<"Fewer than 2 category names specified"<<endl;
                return 1;
            }
            if(!dupname(vargroupvnit->catnames))
            {
                cout<<"Duplicated category name"<<endl;
                return 1;
            }
            vgroupit=++vgroupit;
            if(vgroupit==vargroupfit->end())continue;
        }
        if(*vgroupit=="preds")
        {
            vgroupit=++vgroupit;
            vargroupvnit->preds=parse(*vgroupit,',');
//Check for distinct valid names.
            if(!dupname(vargroupvnit->preds))
            {
                cout<<"Duplicated predictors"<<endl;
                return 1;
            }
            for(prednameit=vargroupvnit->preds.begin();prednameit!=vargroupvnit->preds.end();++prednameit)
            {
                if(!binary_search(varnames.begin(),varnames.end(),*prednameit)
                    &&!binary_search(latnames.begin(),latnames.end(),*prednameit))
                {
                    cout<<"Predictor name not found."<<endl;
                    return 1;
                }
            }
            vgroupit=++vgroupit;
            if(vgroupit==vargroupfit->end())continue;
        }
        vargroupvnit->transform='.';
        if(*vgroupit=="transform")
        {
            vgroupit=++vgroupit;
            vargroupvnit->transform=*vgroupit->begin();
            if(!binary_search(transforms.begin(),transforms.end(),vargroupvnit->transform))
            {
                cout<<"transform symbol not found."<<endl;
                return 1;
            }
            vgroupit=++vgroupit;
            if(vgroupit==vargroupfit->end())continue;
        }
        vargroupvnit->type='.';
        if(*vgroupit=="type")
        {
            vgroupit=++vgroupit;
            vargroupvnit->type=*vgroupit->begin();
            if(!binary_search(types.begin(),types.end(),vargroupvnit->type))
            {
                cout<<"type symbol not found."<<endl;
                return 1;
            }
            vgroupit=++vgroupit;
            if(vgroupit==vargroupfit->end())continue;
        }
    }
//Verify that each vargroup has a type.
    pair<vector<varlocs>::iterator,vector<varlocs>::iterator> typeit;
    typeit=typelookup('.',vargroupvn);
    if(typeit.first<vartab.end())
    {
        cout<<"Variable group without a type."<<endl;
        return 1;
    }
//Now process variables.
    vector<vector<string>>variablef;
    vector<vector<string>>::iterator variablefit;
    vector<string>::iterator vfit,vfitm;
    vector<varlocs>::iterator vf;
    string variable="variable",varn;
    bool typed;
    variablef=keyvar(variable,controlvec);
    if(variablef.empty())
    {
        cout<<"No variables."<<endl;
        return 1;
    }
    vector<string> varlist(variablef.size());
    vector<int> obslist(variablef.size(),0);
    vector<string>::iterator varlistit;
    varlistit=varlist.begin();
    vector<int>::iterator obslistit;
    obslistit=obslist.begin();
    for(variablefit=variablef.begin();variablefit!=variablef.end();++variablefit)
    {
        if(variablefit->size()<4)
        {
            cout<<"variable statement has too few elements."<<endl;
            return 1;
        }
        if(*next(variablefit->end(),-2)!="varname")
        {
            cout<<"varname keyword not found"<<endl;
            return 1;
        }
        varn=*next(variablefit->end(),-1);
        vf=varlookup(varn,vartab);
        if(vf<vartab.end())
        {
            *obslistit=1;
        }
        else
        {
            vf=varlookup(varn,lattab);
            if(vf>=lattab.end())
            {
                cout<<"Variable does not exist"<<endl;
                return 1;
            }
        }
        *varlistit=varn;
        ++varlistit;
        ++obslistit;
    }
    obslistit=obslist.begin();
//Verify that observed responses exit.
    bool respvar=false;
    for(variablefit=variablef.begin();variablefit!=variablef.end();++variablefit)
    {
        typed=false;
        vfit=variablefit->begin();
        vfitm=next(variablefit->end(),-2);
        if(*next(variablefit->end(),-4)=="vargroup")
        {
            vfitm=next(variablefit->end(),-4);
            vargroupvnit=varlookup(*next(variablefit->end(),-3),vargroupvn);
            if(vargroupvnit>=vargroupvn.end())
            {
                cout<<"Variable group does not exist"<<endl;
                return 1;
            }
            vf->transform=vargroupvnit->transform;
            vf->type=vargroupvnit->type;
            typed=true;
            vf->catnames=vargroupvnit->catnames;
            vf->preds=vargroupvnit->preds;
            if(!vf->preds.empty())
            {
                for(prednameit=vf->preds.begin();prednameit!=vf->preds.end();++prednameit)
                {
                    if(!binary_search(varlist.begin(),varlist.end(),*prednameit))
                    {
                        cout<<"Predictor name not found."<<endl;
                        return 1;
                    }
                }
            }
            if(vfit==vfitm)continue;
        }
        if(*vfit=="catnames")
        {
            vfit=++vfit;
            vf->catnames=parse(*vfit,',');
//Check for distinct categories.
            if(vf->catnames.size()<2)
            {
                cout<<"Fewer than 2 category names specified"<<endl;
                return 1;
            }
            if(!dupname(vf->catnames))
            {
                cout<<"Duplicated category name"<<endl;
                return 1;
            }
            vfit=++vfit;
            if(vfit==vfitm)continue;
        }
        if(*vfit=="preds")
        {
            vfit=++vfit;
            vf->preds=parse(*vfit,',');
//Check for distinct valid names.
            if(!dupname(vf->preds))
            {
                cout<<"Duplicated predictors"<<endl;
                return 1;
            }
            if(!vf->preds.empty())
            {
                for(prednameit=vf->preds.begin();prednameit!=vf->preds.end();++prednameit)
                {
                    if(!binary_search(varlist.begin(),varlist.end(),*prednameit))
                    {
                        cout<<"Predictor name not found."<<endl;
                        return 1;
                    }
                }
            }
            vfit=++vfit;
            if(vfit==vfitm)continue;
        }
        if(*vfit=="transform")
        {
           vfit=++vfit;
           vf->transform=*vfit->begin();
           if(!binary_search(transforms.begin(),transforms.end(),vf->transform))
           {
               cout<<"transform symbol not found."<<endl;
               return 1;
           }
           vfit=++vfit;
           if(vfit==vfitm)continue;
        }
        if(*vfit=="type")
        {
            vfit=++vfit;
            vf->type=*vfit->begin();
            if(!binary_search(types.begin(),types.end(),vf->type))
            {
                cout<<"type symbol not found."<<endl;
                return 1;
            }
            vfit=++vfit;
        }
        else
        {
            if(!typed)
            {
                cout<<"Variable has no type"<<endl;
                return 1;
            }
        }
    }
    for(variablefit=variablef.begin();variablefit!=variablef.end();++variablefit)
    {
        if((vf->type=='F'||vf->type=='W'||vf->type=='I')&&*obslistit==0)
        {
            cout<<"Fixed, weight, and ID variables cannot be latent."<<endl;
            return 1;
        }
        if(vf->type!='F'&&vf->type!='W'&&vf->type!='I'&&*obslistit==1)respvar=true;
        ++obslistit;
    }
    if(!respvar)
    {
        cout<<"No responses."<<endl;
        return 1;
    }
//Get observation weights.  Default is all ones for each form.
//Find weight variable.  Default is all weights are 1.
//datasel default has datasel.all as true.
    vec obsweight(nobs,fill::ones);
    vector<int> clustervars(variablef.size()),indvars(variablef.size());
    xsel  datasel;
    datasel.all=true;
    pair<vector<varlocs>::iterator,vector<varlocs>::iterator> typewit;
    typewit=typelookup('W',vartab);
    vector<int>::iterator formit,posit;
    if(typewit.first<vartab.end())
    {
       if(typewit.second<vartab.end())
       {
           cout<<"Only one weight variable permitted."<<endl;
           return 1;
       }
       posit=typewit.first->positions.begin();
       for(formit=typewit.first->forms.begin();formit!=typewit.first->forms.end();++formit)
       {
           j=*formit;
           k=*posit;
           datafit=next(dataf.begin(),j);
           obsweight.subvec(cumobs(j)-formcount(j),cumobs(j)-1)=datafit->vars.col(k);
           ++posit;
       }
       if(!obsweight.is_finite())
       {
           cout<<"Weight variable not always finite."<<endl;
           return 1;
       }
       if(min(obsweight)<0)
       {
           cout<<"Weight variable has negative values."<<endl;
           return 1;
       }
       if(max(obsweight)==0)
       {
           cout<<"Weight variable always 0."<<endl;
           return 1;
       }
       if(min(obsweight)==0)
       {
           datasel.all=false;
           datasel.list.set_size(nobs);
           j=0;
           for(i=0;i<nobs;i++)
           {
               if(obsweight(i)>0)
               {
                   datasel.list(j)=i;
                   j++;
               }
           }
           datasel.list.resize(j);
       }
    }
//Get individual identifier.
    ivec ID=regspace<ivec>(0,nobs-1);
    pair<vector<varlocs>::iterator,vector<varlocs>::iterator> typeiit;
    typeiit=typelookup('I',vartab);
    if(typeiit.first<vartab.end())
    {
       if(typeiit.second<vartab.end())
       {
          cout<<"Only one weight variable permitted."<<endl;
          return 1;
       }
       posit=typeiit.first->positions.begin();
       for(formit=typeiit.first->forms.begin();formit!=typeiit.first->forms.end();++formit)
       {
          j=*formit;
          k=*posit;
          datafit=next(dataf.begin(),j);
          ID.subvec(cumobs(j)-formcount(j),cumobs(j)-1)=conv_to<ivec>::from(datafit->vars.col(k));
          ++posit;
       }
       if(!ID.is_finite())
       {
          cout<<"ID number not always finite."<<endl;
          return 1;
       }
       if(nobs-size(unique(ID))[0]>0)
       {
          cout<<"ID numbers not unique."<<endl;
       }
    }
    cout<<"end"<<endl;
    return 0;
}
