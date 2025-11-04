*//General model for maximum likelihood for latent structures.
//Multiple forms are possible, and missing data
//may exist.  All data variables are read as double-precision numbers. If a variable entry is blank or not  //numeric, then it has value NaN.
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
//    There is at most one latent statement.  This statement provides the names of the latent variables with the keyword
//    variables.  Variablre names follow in comma-separated form.
//
//vargroup
//    A vargroup statement provides a template for one or more variables used in analysis.
//    The statement provides elements of a vardef struct that are common to these variables.
//    Common elements may include catnames, constant, constant_list, full, full_list, o, preds, transform, and type.
//    Definitions of transform and type are as in genresp.cpp,
//    except that type='W' corresponds to a weight variable and type='I' corresponds to a numerical identifier for an
//    observation.  catnames are value names for a polytomous variable in comma-separated format. constant may equal
//    true or false, with true for constant predictors for all variables and false otherwise.  If constant is false,
//    then constant_list is equal to a comma-separated list of integers that correspond to variables in the varname list
//    for which constants are used.  The default is constant true and constant_list empty. deg1
//    may equal true or false, with true for  models with one parameter per predictor per variable predicted
//    and false otherwise.  If deg1 is false,
//    then deg1_list is equal to a comma-separated list of integers that correspond to variables in the varname list
//    for which degree-one models are used.
//    The default is deg1 true and deg1_list empty.  fullpred may equal true or false,
//    with true for using the predictors in preds for all variables predicted
//    and false otherwise.  If fullpred is false,
//    then fullpred_list is equal to a comma-separated list of integers that correspond to variables in the varname list
//    for which the predictors in preds are used.  The default is usepred true and usepred_list empty.
//    The keyword o is equal to
//    a comma-separated list of double values that correspond to values of the offset vector.
//    preds are predicting variables in comma-separated format.  In addition,
//    vargroupname is the name for the group and varsize is the number of variable names specified by varname for a
//    variable in the group.  Thia number is 1 except for the type T (truncated), where it is 2 abd for types E
//    (Dirichlet) and R (rank logit).
//
//variable
//    A variable statement defines a variable that corresponds to one or more observed or latent variables.  The
//    format corresponds to the format of the vargroup statement; however, varname provides a comma-separated list of
//    the original one-dimensional variables that are the components of the variable, and vardefname provides a name of the
//    variable if not the same as the first member of varname.  Here varname is used for underlying one-dimensional
//    variables.  Either vargroupname specifies the variable group and only varname and vardefname are specified or
//    all specifications for a variable group appear except for varsize,
//
//method
//   The algorithm to use.  The keyword algorithm can equal G for gradient ascent, C for conjugate gradient ascent,
//   N for modified Newton-Raphson, and L for Louis approximation.  The default is N.
//
//parameters
//   parameters gives any algorithm parameters tha do not have their default values.  For definitions,
//   see convergence.pdf.  Keywords are as follows:
//   eta is the eta parameter that must be positive and less than 1.  The default value is 0.5.
//   gamma1 is the gamma1 parameter that must exceed 1 and has default value 2.0.
//   gamma2 is the gamma2 parameter that must be positive and less than 1 and has default value 0.1.
//   kappa is the kappa  parameter that must be positive and has default value 3.0.
//   maxit is the positive maximum number of main iterations.  The default value is 100.
//   maxita is the positive maximum number of iterations for a line search.  The default value is 10.
//   tol is the tolerance for adequate convergence for a main iteration.  The default value is 0.001.
//
//adapt
//   adapt specifies continuous latent variables for which adaptive quadrature is used.
//   If adapt is omitted, then adaptive quadature is not used.
//   If adaptall is true, then all linear and quadratic terms are used for all continuous latent variables.
//   If linear is associate with a comma-separated list of names of
//   If allquad is true, then all quadratic terms are used for continuous variables.
//   If a


//If adapt is not present, then no adaptive quadrature is used.
//

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


//
//quad qu
//   qu is G for Gauss-Hermite quadrature and Q for normal quantiles.
//   The default is G.
//  
//points n
//   n is the number of quadrature points per dimension.  The default values are
//   9 for one dimension, 7 for two dimensions, 5 for three dimensions, and 3 otherwise.



#include<armadillo>
#include<string>
#include<set>
using namespace std;
using namespace arma;
//Select elements of vector.  all indicates all elements.  list lists elements.
struct xsel
{
    bool all;
    uvec list;
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
    vector<int> forms;
    vector<int> positions;
    string varname;
};
struct vardef
{
    vector<string>catnames;
    xsel constant;
    xsel deg1;
    xsel fullpred;
    vec o;
    bool obs;
    vector<vector<varlocs>::const_iterator> predits;
    vector<string> preds;
    mat predweights;
    char transform;
    char type;
    string vardefname;
    vector<string>varname;
    vector<vector<varlocs>::const_iterator> varnameit;
};
typedef tuple<vector<string>,vector<int>,vector<int>,vector<int>,vector<double>,
    bool,vector<string>,char,char,int> vdeftuple;
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
vector<varlocs>::const_iterator varlookup(const string & , const vector<varlocs> & );
bool vdefsort(const vardef & a,const vardef & b);
vector<vardef>::const_iterator vardeflookup(const string & , const vector<vardef> & );
vector<varlocs> vfp(const vector<vn> & );
vector<string>parse(const string & , const char & );
bool dupname(const vector<string> & );
bool varlocsort(const varlocs & a,const varlocs & b);
bool varlocchk(const varlocs & a,const varlocs & b)
    {return(a.varname==b.varname);};
vector<int> xseltovector(const xsel & );
vector<double>vectovector(const vec & );
vdeftuple tuplevardef(const vardef & a){vector<int> ac,ad,af;vector<double> av;ac=xseltovector(a.constant);
    ad=xseltovector(a.deg1);af=xseltovector(a.fullpred); av=vectovector(a.o);return make_tuple(a.catnames,ac,ad,af,
    av,a.obs,a.preds,a.transform,a.type,a.varname.size());};
vector<pair<vector<int>,int>>obspat(const vector<vn> & , const vector<varlocs> & , const vector<vardef> & );
int intsel(const xsel & , const int & );
int sintsel(const xsel & , const int & );
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
//trn is a transform and ty is a type.
    char trn, ty;
//nc is number of control records.
//ncats is number of categories.
//ncols is number of columns of x.
//npars is number of parameters of response variable.
//npred is number of predictors.
//nvars is number of variables
//nvd is number of observed variables.
    int h, i, j, k, m, nc, ncats, ncols, npars, npreds, nvars, nvd ;
    nc=control.n_rows;
    vector<vector<string>>controlvec(nc);
    controlvec=keysort(control);
    if(controlvec.empty())return 1;
//Check for duplicates.
    if(nc>1)
    {
        if(adjacent_find(controlvec.begin(),controlvec.end())<controlvec.end())
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
    vector<varlocs>::const_iterator cvartabit, cvartabit1;
    vector<varlocs>::iterator vartabit;
    vartab=vfp(dataf);
    if(vartab.empty())return 1;
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
    vector<varlocs>lattab, vltab;
    vector<string> latnames;
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
        if(adjacent_find(latnames.begin(),latnames.end())<latnames.end())
        {
            cout<<"Duplicate latent variable."<<endl;
            return 1;
        }
//Set up latent variables in lattab.
        lattab.resize(latnames.size());
        vartabit=lattab.begin();
        for(latnamesit=latnames.begin();latnamesit!=latnames.end();++latnamesit){vartabit->varname=*latnamesit;++vartabit;}
//Merge variablels.
        vltab.resize(vartab.size()+lattab.size());
        merge(vartab.begin(),vartab.end(),lattab.begin(),lattab.end(),vltab.begin(),varlocsort);
//Check for duplicates.
        if(adjacent_find(vltab.begin(),vltab.end(),varlocchk)<vltab.end())
        {
            cout<<"Duplicate variable."<<endl;
            return 1;
        }
    }
    else
    {
        vltab.resize(vartab.size());
        copy(vartab.begin(),vartab.end(),vltab.begin());
    }
//Clean up.
    vartab.clear();
    lattab.clear();
    latnames.clear();
//Set up variable groups.
    vector<vector<string>>vargroupf;
    vector<vector<string>>::iterator vargroupfit;
    vector<string>::iterator vgroupit;
    string vargroup="vargroup",varn;
    vargroupf=keyvar(vargroup,controlvec);
    vector<vardef> vargroupvn(vargroupf.size());
    vector<vardef>::iterator vargroupvnit;
    vector<vardef>::const_iterator cvargroupvnit;
    vector<string> conlist, predname;
    vector<string>::iterator conlistit, prednameit;
    uvec::iterator uvecit;
    vec::iterator vecit;
    vector<char>transforms={'G','H','L','N'};
    vector<char>types={'B','C','D','E','F','G','H','I','L','M','P','R','S','T','W'};
    vargroupvnit=vargroupvn.begin();
    for(vargroupfit=vargroupf.begin();vargroupfit!=vargroupf.end();++vargroupfit)
    {
        if(vargroupfit->size()<4)
        {
            cout<<"Vargroup statement does not have sufficient elements."<<endl;
            return 1;
        }
        vgroupit=vargroupfit->begin();
        vargroupvnit->constant.all=true;
        vargroupvnit->deg1.all=true;
        vargroupvnit->fullpred.all=true;
        if(*vgroupit=="catnames")
        {
            ++vgroupit;
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
            ++vgroupit;
            if(vgroupit==vargroupfit->end())
            {
                cout<<"No vargroupname."<<endl;
                return 1;
            }
        }
//Look for constant specification.
        if(*vgroupit=="constant")
        {
            ++vgroupit;
            if(*vgroupit=="false")vargroupvnit->constant.all=false;
            ++vgroupit;
            if(vgroupit==vargroupfit->end())
            {
                cout<<"No vargroupname."<<endl;
                return 1;
            }
        }
//Look for constant_list.
        if(*vgroupit=="constant_list")
        {
            ++vgroupit;
            conlist=parse(*vgroupit,',');
            if(!conlist.empty()){
                vargroupvnit->constant.list.set_size(conlist.size());
                uvecit=vargroupvnit->constant.list.begin();
                for(conlistit=conlist.begin();conlistit!=conlist.end();++conlistit){
                    try{*uvecit=stoi(*conlistit);}
                    catch(...){cout<<"Bad integer in vargroup list specification."<<endl; return 1;}
                    if(*uvecit<0){
                        cout<<"Negative integer in vargroup list specification."<<endl;
                        return 1;
                    }
                    ++uvecit;
                }
            }
            ++vgroupit;
            if(vgroupit==vargroupfit->end())
            {
                cout<<"No vargroupname."<<endl;
                return 1;
            }
        }
//Look for degree one specification.
        if(*vgroupit=="deg1")
        {
            ++vgroupit;
            if(*vgroupit=="false")vargroupvnit->deg1.all=false;
            ++vgroupit;
            if(vgroupit==vargroupfit->end())
            {
                cout<<"No vargroupname."<<endl;
                return 1;
            }
        }
//Look for deg1_list.
        if(*vgroupit=="deg1_list")
        {
            ++vgroupit;
            conlist=parse(*vgroupit,',');
            if(!conlist.empty()){
                vargroupvnit->deg1.list.set_size(conlist.size());
                uvecit=vargroupvnit->deg1.list.begin();
                for(conlistit=conlist.begin();conlistit!=conlist.end();++conlistit){
                    try{*uvecit=stoi(*conlistit);}
                    catch(...){cout<<"Bad integer in vargroup list specification."<<endl; return 1;}
                    if(*uvecit<0){
                        cout<<"Negative integer in vargroup list specification."<<endl;
                        return 1;
                    }
                    ++uvecit;
                }
            }
            ++vgroupit;
            if(vgroupit==vargroupfit->end())
            {
                cout<<"No vargroupname."<<endl;
                return 1;
            }
        }
//Look for fullpred case.
        if(*vgroupit=="fullpred")
        {
            ++vgroupit;
            if(*vgroupit=="false")vargroupvnit->fullpred.all=false;
            ++vgroupit;
            if(vgroupit==vargroupfit->end())
            {
                cout<<"No vargroupname."<<endl;
                return 1;
            }
        }
//Look for fullpred_list.
        if(*vgroupit=="fullpred_list")
        {
            ++vgroupit;
            conlist=parse(*vgroupit,',');
            if(!conlist.empty()){
                vargroupvnit->fullpred.list.set_size(conlist.size());
                uvecit=vargroupvnit->fullpred.list.begin();
                for(conlistit=conlist.begin();conlistit!=conlist.end();++conlistit){
                    try{*uvecit=stoi(*conlistit);}
                    catch(...){cout<<"Bad integer in vargroup list specification."<<endl; return 1;}
                    if(*uvecit<0){
                        cout<<"Negative integer in vargroup list specification."<<endl;
                        return 1;
                    }
                    ++uvecit;
                }
            }
            ++vgroupit;
            if(vgroupit==vargroupfit->end())
            {
                cout<<"No vargroupname."<<endl;
                return 1;
            }
        }
//Constant value.
        if(*vgroupit=="o"){
            ++vgroupit;
            conlist=parse(*vgroupit,',');
            if(conlist.empty()){
                cout<<"Constant vector must have elements."<<endl;
                return 1;
            }
            vargroupvnit->o.set_size(conlist.size());
            vecit=vargroupvnit->o.begin();
            for(conlistit=conlist.begin();conlistit!=conlist.end();++conlistit){
                try{*vecit=stod(*conlistit);}
                catch(...){cout<<"Bad integer in vargroup list specification."<<endl; return 1;}
                ++vecit;
            }
            ++vgroupit;
            if(vgroupit==vargroupfit->end())
            {
                cout<<"No vargroupname."<<endl;
                return 1;
            }
        }
        if(*vgroupit=="preds")
        {
            ++vgroupit;
            vargroupvnit->preds=parse(*vgroupit,',');
//Check for distinct valid names.
            if(!dupname(vargroupvnit->preds))
            {
                cout<<"Duplicated predictors"<<endl;
                return 1;
            }
            ++vgroupit;
            if(vgroupit==vargroupfit->end())
            {
                cout<<"No vargroupname."<<endl;
                return 1;
            }
        }
        vargroupvnit->transform='.';
        if(*vgroupit=="transform")
        {
            ++vgroupit;
            vargroupvnit->transform=*vgroupit->begin();
            if(!binary_search(transforms.begin(),transforms.end(),vargroupvnit->transform))
            {
                cout<<"transform symbol not found."<<endl;
                return 1;
            }
            ++vgroupit;
            if(vgroupit==vargroupfit->end())
            {
                cout<<"No vargroupname."<<endl;
                return 1;
            }
        }
        if(*vgroupit=="type")
        {
            ++vgroupit;
            vargroupvnit->type=*vgroupit->begin();
            if(!binary_search(types.begin(),types.end(),vargroupvnit->type))
            {
                cout<<"type symbol not found."<<endl;
                return 1;
            }
            vargroupvnit->varname.resize(1);
            vargroupvnit->varname[0]="0";
            if(vargroupvnit->type=='T'){
                vargroupvnit->varname.resize()=2;
                vargroupvnit->varname[1]={"1"};
            }
        }
        else
        {
            cout<<"Missing type in vargroup statement."<<endl;
            return 1;
        }
        ++vgroupit;
        if(vgroupit==vargroupfit->end())
        {
            cout<<"No vargroupname."<<endl;
            return 1;
        }
        if(*vgroupit!="vargroupname")
        {
            cout<<"No vargroupname."<<endl;
        }
        ++vgroupit;
        vargroupvnit->vardefname=*vgroupit;
        ++vgroupit;
        if(vargroupvnit->type=='E'||vargroupvnit->type=='R'){
            if(*vargroupit!="varsize")
            {
                cout<<"varsize keyword not found"<<endl;
                return 1;
            }
            ++vgroupit;
            
            vargroupvnit->varname=parse(varn,',');
            if(vargroupvnit->varname.size()<2)
            {
                cout<<"Variable list too small for type."<<endl;
                return 1;
            ++vgroupit
        }
        if(vgroupit!=vargroupfit->end()){
            cout<<"Extra material on vargroup statement."<<endl;
            return 1;
        }
    }
    sort(vargroupvn.begin(),vargroupvn.end(),vdefsort);
//Now process variables.
    vector<vector<string>>variablef;
    vector<vector<string>>::iterator variablefit;
    vector<string>::iterator vfit,vfitm;
    string variable="variable";
    vector<string>varns;
    bool respvar=false;
    variablef=keyvar(variable,controlvec);
    if(variablef.empty())
    {
        cout<<"No variables."<<endl;
        return 1;
    }
//Which variables are to be used.
    vector<vardef> vardefs(variablef.size());
    vector<string> varlist(vltab.size());
    vector<vardef>::iterator vardefit, vardefit1;
    vector<vector<varlocs>::const_iterator>::iterator varnit;
    vardefit=vardefs.begin();
    vector<string>::iterator varlistit;
    varlistit=varlist.begin();
    nvd=0;
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
        vardefit->varname=parse(varn,',');
        if(vardefit->varname.empty())
        {
            cout<<"Empty variable list."<<endl;
            return 1;
        }
//Tentative variable name.
        vardefit->vardefname=vardefit->varname[0];
        vardefit->varnameit.resize(vardefit->varname.size());
        varnit=vardefit->varnameit.begin();
        for(string vnm:vardefit->varname)
        {
            cvartabit=varlookup(vnm,vltab);
            if(cvartabit>=vltab.cend())
            {
                cout<<"Variable does not exist."<<endl;
                return 1;
            }
            if(vnm==vardefit->varname[0])
            {
                cvartabit1=cvartabit;
                vardefit->obs=true;
                if(cvartabit->forms.empty())vardefit->obs=false;
            }
            else
            {
                if(cvartabit->forms.size()!=cvartabit1->forms.size())
                {
                    cout<<"Variables do not appear in the same forms."<<endl;
                    return 1;
                }
                if(!cvartabit->forms.empty())
                {
                    if(!equal(cvartabit->forms.begin(),cvartabit->forms.end(),cvartabit1->forms.begin()))
                    {
                        cout<<"Variables do not appear in the same forms."<<endl;
                        return 1;
                    }
                }
            }
            if(vardefit->obs)nvd++;
            *varnit=cvartabit;
            ++varnit;
            *varlistit=vnm;
            ++varlistit;
            ++vardefit;
        }
    }
//Check for duplicate names.
    m=distance(varlist.begin(),varlistit);
    varlist.resize(m);
    sort(varlist.begin(),varlist.end());
    if(adjacent_find(varlist.begin(),varlist.end())<varlist.end())
    {
        cout<<"Duplicate variables."<<endl;
        return 1;
    }
    vardefit=vardefs.begin();
//Finish vardef specifications.
    for(variablefit=variablef.begin();variablefit!=variablef.end();++variablefit)
    {
        vardefit->transform='.';
        vardefit->type='.';
        vardefit->constant.all=true;
        vardefit->deg1.all=true;
        vardefit->fullpred.all=true;
        vfit=variablefit->begin();
        vfitm=next(variablefit->end(),-2);
        if(*next(variablefit->end(),-4)=="vargroupname")
        {
            vfitm=next(variablefit->end(),-4);
            cvargroupvnit=vardeflookup(*next(variablefit->end(),-3),vargroupvn);
            if(cvargroupvnit>=vargroupvn.cend())
            {
                cout<<"Variable group does not exist"<<endl;
                return 1;
            }
            vardefit->transform=cvargroupvnit->transform;
            vardefit->type=cvargroupvnit->type;
            vardefit->catnames=cvargroupvnit->catnames;
            vardefit->preds=cvargroupvnit->preds;
            vardefit->constant.all=cvargroupvnit->constant.all;
            vardefit->deg1.all=cvargroupvnit->deg1.all;
            vardefit->fullpred.all=cvargroupvnit->fullpred.all;
            vardefit->constant.list=cvargroupvnit->constant.list;
            if(!vardefit->constant.list.empty()){
                if(vardefit->constant.list.max()>=vardefit->varname.size()){
                    cout<<"Excessive integer in variable list."<<endl;
                    return 1;
                }
            }
            vardefit->deg1.list= cvargroupvnit->deg1.list;
            vardefit->fullpred.list=cvargroupvnit->fullpred.list;
            vardefit->o=cvargroupvnit->o;
            if(vfit==vfitm)
            {
                ++vardefit;
                continue;
            }
        }
        if(*vfit=="catnames")
        {
            ++vfit;
            vardefit->catnames=parse(*vfit,',');
//Check for distinct categories.
            if(vardefit->catnames.size()<2)
            {
                cout<<"Fewer than 2 category names specified"<<endl;
                return 1;
            }
            if(!dupname(vardefit->catnames))
            {
                cout<<"Duplicated category name"<<endl;
                return 1;
            }
            ++vfit;
            if(vfit==vfitm)
            {
                ++vardefit;
                continue;
            }
        }
//Look for constant specification.
        if(*vfit=="constant")
        {
            ++vfit;
            if(*vfit=="false")vardefit->constant.all=false;
            ++vfit;
            if(vfit==vfitm)
            {
                ++vardefit;
                continue;
            }
        }
//Look for constant_list.
        if(*vfit=="constant_list")
        {
            ++vfit;
            conlist=parse(*vfit,',');
            vardefit->constant.list.set_size(conlist.size());
            if(!conlist.empty()){
                uvecit=vardefit->constant.list.begin();
                for(conlistit=conlist.begin();conlistit!=conlist.end();++conlistit){
                    try{*uvecit=stoi(*conlistit);}
                    catch(...){cout<<"Bad integer in variable list specification."<<endl; return 1;}
                    if(*uvecit<0){
                        cout<<"Negative integer in variable list specification."<<endl;
                        return 1;
                    }
                    if(*uvecit>=vardefit->varname.size()){
                        cout<<"Excessive integer in variable list specification."<<endl;
                        return 1;
                    }
                    ++uvecit;
                }
            }
            ++vfit;
            if(vfit==vfitm)
            {
                ++vardefit;
                continue;
            }
        }
//Look for degree one specification.
        if(*vfit=="deg1")
        {
            ++vfit;
            if(*vfit=="false")vardefit->deg1.all=false;
            ++vfit;
            if(vfit==vfitm)
            {
                ++vardefit;
                continue;
            }
        }
//Look for deg1_list.
        if(*vfit=="deg1_list")
        {
            ++vfit;
            conlist=parse(*vfit,',');
            vardefit->deg1.list.set_size(conlist.size());
            if(!conlist.empty()){
                vardefit->deg1.list.set_size(conlist.size());
                uvecit=vardefit->deg1.list.begin();
                for(conlistit=conlist.begin();conlistit!=conlist.end();++conlistit){
                    try{*uvecit=stoi(*conlistit);}
                    catch(...){cout<<"Bad integer in variable list specification."<<endl; return 1;}
                    if(*uvecit<0){
                        cout<<"Negative integer in variable list specification."<<endl;
                        return 1;
                    }
                    if(*uvecit>=vardefit->varname.size()){
                        cout<<"Excessive integer in variable list specification."<<endl;
                        return 1;
                    }
                    ++uvecit;
                }
            }
            ++vfit;
            if(vfit==vfitm)
            {
                ++vardefit;
                continue;
            }
        }
//Look for fullpred case.
        if(*vfit=="fullpred")
        {
            ++vfit;
            if(*vfit=="false")vardefit->fullpred.all=false;
            ++vfit;
            if(vfit==vfitm)
            {
                ++vardefit;
                continue;
            }
        }
//Look for fullpred_list.
        if(*vfit=="fullpred_list")
        {
            ++vfit;
            conlist=parse(*vfit,',');
            vardefit->fullpred.list.set_size(conlist.size());
            if(!conlist.empty()>0){
                vardefit->fullpred.list.set_size(conlist.size());
                uvecit=vardefit->fullpred.list.begin();
                for(conlistit=conlist.begin();conlistit!=conlist.end();++conlistit){
                    try{*uvecit=stoi(*conlistit);}
                    catch(...){cout<<"Bad integer in vargroup specification."<<endl; return 1;}
                    if(*uvecit<0){
                        cout<<"Negative integer in vargroup specification."<<endl;
                        return 1;
                    }
                    if(*uvecit>=vardefit->varname.size()){
                        cout<<"Excessive integer in variable list specification."<<endl;
                        return 1;
                    }
                    ++uvecit;
                }
            }
            ++vfit;
            if(vfit==vfitm)
            {
                ++vardefit;
                continue;
            }
        }
//Constant value.
        if(*vfit=="o"){
            ++vfit;
            conlist=parse(*vfit,',');
            if(conlist.empty()){
                cout<<"Constant vector must have elements."<<endl;
                return 1;
            }
            vardefit->o.set_size(conlist.size());
            vecit=vargroupvnit->o.begin();
            for(conlistit=conlist.begin();conlistit!=conlist.end();++conlistit){
                try{*vecit=stod(*conlistit);}
                catch(...){cout<<"Bad integer in vargroup list specification."<<endl; return 1;}
                ++vecit;
            }
            ++vfit;
            if(vfit==vfitm)
            {
                ++vardefit;
                continue;
            }
        }
        if(*vfit=="preds")
        {
            ++vfit;
            vardefit->preds=parse(*vfit,',');
//Check for distinct valid names.
            if(!dupname(vardefit->preds))
            {
                cout<<"Duplicated predictors"<<endl;
                return 1;
            }
            ++vfit;
            if(vfit==vfitm)
            {
                ++vardefit;
                continue;
            }
        }
        if(*vfit=="transform")
        {
            ++vfit;
            vardefit->transform=*vfit->begin();
            if(!binary_search(transforms.begin(),transforms.end(),vardefit->transform))
            {
                cout<<"transform symbol not found."<<endl;
                return 1;
            }
            ++vfit;
            if(vfit==vfitm)
            {
                ++vardefit;
                continue;
            }
        }
        if(*vfit=="type")
        {
            ++vfit;
            vardefit->type=*vfit->begin();
            if(!binary_search(types.begin(),types.end(),vardefit->type))
            {
                cout<<"type symbol not found."<<endl;
                return 1;
            }
            ++vfit;
            if(vfit==vfitm)
            {
                ++vardefit;
                continue;
            }
        }
        if(*vfit=="vardefname")
        {
            ++vfit;
            vardefit->vardefname=*vfit;
        }
        ++vardefit;
    }
    sort(vardefs.begin(),vardefs.end(),vdefsort);
//ID flag, obsweight flag, and observed variable flag.
    bool idfl=false, obsfl=false, obfl;
    vector<vardef>::iterator IDit,obsweightit;
    vector<varlocs>::const_iterator IDvit,obsvit;
    IDit=vardefs.end();
    obsweightit=vardefs.end();
    int patsize=0;
//Check on predictors.
    for(vardefit=vardefs.begin();vardefit!=vardefs.end();++vardefit)
    {
        if(!vardefit->preds.empty())
        {
            vardefit->predits.resize(vardefit->preds.size());
            varnit=vardefit->predits.begin();
            for(prednameit=vardefit->preds.begin();prednameit!=vardefit->preds.end();++prednameit)
            {
                if(!binary_search(varlist.begin(),varlist.end(),*prednameit))
                {
                    cout<<"Predictor not in variable list."<<endl;
                    return 1;
                }
                cvartabit=varlookup(*prednameit,vltab);
                *varnit=cvartabit;
                ++varnit;
            }
        }
//Check for special types that can only occur once and check other rules on types.
        if(vardefit->type=='.')
        {
            cout<<"Variable has no type"<<endl;
            return 1;
        }
        if((vardefit->type=='F'||vardefit->type=='W'||vardefit->type=='I')
           &&(!vardefit->obs||vardefit->varname.size()>1))
        {
            cout<<"Fixed, weight, and ID variables cannot be latent or multidimensional."<<endl;
            return 1;
        }
        if(vardefit->type!='E'&vardefit->type!='R'&vardefit->type!='T'&vardefit->varname.size()>1){
            cout<<"Type and number of variables inconsistent."<<endl;
            return 1;
        }
        if(vardefit->type=='T'&vardefit->varname.size()!=2){
            cout<<"Type and number of variables inconsistent."<<endl;
            return 1;
        }
        if(vardefit->type!='F'&vardefit->type!='W'&vardefit->type!='I')patsize++;
        if(vardefit->type=='I')
        {
            if(IDit<vardefs.end())
            {
                cout<<"Only one ID is allowed."<<endl;
                return 1;
            }
            IDit=vardefit;
            IDvit=IDit->varnameit[0];
            idfl=true;
            nvd--;
        }
        if(vardefit->type=='W')
        {
            if(obsweightit<vardefs.end())
            {
                cout<<"Only one weight is allowed."<<endl;
                return 1;
            }
            obsweightit=vardefit;
            obsvit=obsweightit->varnameit[0];
            obsfl=true;
            nvd--;
        }
//Observed response variables.
        if(vardefit->type!='W'&&vardefit->type!='I'&&vardefit->type!='F'&&vardefit->obs)respvar=true;
    }
    if(!respvar)
    {
        cout<<"No responses."<<endl;
        return 1;
    }
//Find patterns.
    vector<vdeftuple>vdeftuples(patsize);
    vector<vdeftuple>::iterator vdeftuplesit;
    vdeftuplesit=vdeftuples.begin();
    for(vardefit=vardefs.begin();vardefit!=vardefs.end();++vardefit)
    {
        if((vardefit->type!='F')&(vardefit->type!='W')&(vardefit->type!='I')){
            *vdeftuplesit=tuplevardef(*vardefit);
            ++vdeftuplesit;
        }
    }
//    sort(vdeftuples.begin(),vdeftuples.end());
//vdeftuplesit=unique(vdeftuples.begin(),vdeftuples.end());
    h=distance(vdeftuples.begin(),vdeftuplesit);
    vdeftuples.resize(h);
    field<pattern>  patterns(h);
    vdeftuplesit=vdeftuples.begin();
    for(i=0;i<h;i++){
        trn=get<7>(*vdeftuplesit);
        patterns(i).choice.transform=trn;
        ty=get<8>(*vdeftuplesit);
        patterns(i).choice.type=ty;
        ncats=(get<0>(*vdeftuplesit)).size();
        nvars=get<9>(*vdeftuplesit);
        npreds=(get<6>(*vdeftuplesit)).size();
        patterns(i).o=get<4>(*vdeftuplesit);
        obfl=get<5>(*vdeftuplesit);
/*
 param by pred
 const yes or no
 
 */
        switch(ty){
            case 'B':
                npars=2;
                ncols=npreds+1;
                break;
            case 'C':
                npars=ncats-1;
                ncols=npars+npreds;
                break;
            case 'D':
                npars=2;
                ncols=npars+npreds;
                break;
            case 'E':
                npars=nvars;
                break;
            case 'G':
                npars=ncats-1;
                break;
            case 'H':
                npars=2;
                break;
            case 'L':
                npars=ncats-1;
                break;
            case 'M':
                npars=2;
                break;
            case 'R':
                npars=ncats-1;
                break;
            case 'S':
                npars=1;
                ncols=0;
                
                break;
            case 'T':
                npars=2;
        }
        ncols=npars+npreds;
        patterns(i).o.set_size(npars);
        patterns(i).o.zeros();
        patterns(i).x.set_size(npars,ncols);
    //patterns(i),x,submat
//        patterns(i).c;
        ++vdeftuplesit;
    }
        
//Find weight variable.  Default is all weights are 1.
//datasel default has datasel.all as true.
    vec obsweight(nobs,fill::ones);
    xsel  datasel;
    datasel.all=true;
    vector<int>::const_iterator formit,posit;
    if(obsfl)
    {
        posit=obsvit->positions.cbegin();
        for(formit=obsvit->forms.cbegin();formit!=obsvit->forms.cend();++formit)
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
    if(idfl)
    {
       posit=IDvit->positions.begin();
       for(formit=IDvit->forms.begin();formit!=IDvit->forms.end();++formit)
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
       if(nobs>size(unique(ID))[0])
       {
          cout<<"ID numbers not unique."<<endl;
       }
    }
//Check on missing data.
    vector<pair<vector<int>,int>>patdata(nobs),patex(nobs);
    vector<pair<vector<int>,int>>::iterator patdatait,patexit;
    patdata=obspat(dataf,vltab,vardefs);
//Get number of patterns and map observations to patterns.
    sort(patdata.begin(),patdata.end());
    vector<int>patgroups(nobs);
    vector<int>::iterator patgroupsit;
    patdatait=patdata.begin();
    patexit=patex.begin();
    patgroupsit=patgroups.begin();
    *patgroupsit=0;
    patexit->first.resize(patdatait->first.size());
    copy(patdatait->first.begin(),patdatait->first.end(),patexit->first.begin());
    patexit->second=patdatait->second;
    xsel patno;
    patno.list.set_size(nobs);
    patno.all=false;
    j=0;
//Flag for repeat patterns.
    bool patorder=true;
    for(patdatait=next(patdata.begin(),1);patdatait!=patdata.end();++patdatait)
    {
        k=patdatait->second;
        if(!equal(patdatait->first.begin(),patdatait->first.end(),
            patexit->first.begin(),patexit->first.end()))
        {
            ++patexit;
            j++;
            patexit->first.resize(patdatait->first.size());
            copy(patdatait->first.begin(),patdatait->first.end(),patexit->first.begin());
            patexit->second=patdatait->second;
        }
        patno.list(k)=j;
        if(j!=k)patorder=false;
    }
    patex.resize(j+1);

//All patterns are distinct.
    if(patorder)
    {
        patno.all=true;
        patno.list.reset();
    }
//Only last pattern can be all observed variables.
    field<xsel>  patternnumber(j+1);
    if(j>0)for(i=0;i<j;i++)patternnumber(i).all=false;
    if(count(patex[j].first.begin(),patex[j].first.end(),1)==nvd)patternnumber(j).all=true;
//Set up data.
    field<field<resp>> data(nobs);
    for(i=0;i<nobs;i++)
    {
        j=intsel(patno,i);
        h=sintsel(patternnumber(j),patsize);
        data(i).resize(h);
    }

//Get algorithm. Default is Newton-Raphson.
    vector<vector<string>>algos;
    string algo="algorithm",method="method";
    algos=keyvar(method,controlvec);
    char algorithm='N';
    if(algos.size()>1)
    {
        cout<<"Extra method statement."<<endl;
        return 1;
    }
    if(!algos.empty())
    {
        if(algos[0][0]!="algorithm")
        {
            cout<<"Missing keyword for algorithm."<<endl;
        }
        switch (algos[0][1][0])
        {
            case 'c':
            {
                algorithm='C';
                break;
            }
            case 'C':
            {
                algorithm='C';
                break;
            }
            case 'g':
            {
                algorithm='G';
                break;
            }
            case 'G':
            {
                algorithm='G';
                break;
            }
            case 'l':
            {
                algorithm='L';
                break;
            }
            case 'L':
            {
                algorithm='L';
                break;
            }
            case 'n':
            {
                algorithm='N';
                break;
            }
            default:
            {
                algorithm='N';
            }
        }
    }
//Algorithm specifications.
    params mparams;
    mparams.print=true;
    mparams.maxit=100;
    mparams.maxits=10;
    mparams.eta=0.5;
    mparams.gamma1=2.0;
    mparams.gamma2=0.1;
    mparams.kappa=3.0;
    mparams.tol=0.001;
//Find convergence tolerance.  Default is 0.001.
    vector<vector<string>> parameters;
    vector<vector<string>>::iterator parametersit;
    vector<string>::iterator parametersit1;
    vector<string>::iterator parametersitm;
    string param="parameters",to,tol="tolerance";
    parameters=keyvar(param,controlvec);
    if(parameters.size()>1)
    {
        cout<<"Extra parameters statement."<<endl;
        return 1;
    }
    if(!parameters.empty())
    {
        parametersit=parameters.begin();
        parametersit1=parametersit->begin();
        parametersitm=parametersit->end();
        while(parametersit1<parametersitm)
        {
            if(*parametersit1=="eta")
            {
               ++parametersit1;
               try{mparams.eta=stod(*parametersit1);}
               catch(...){mparams.eta=0.5;}
               if(mparams.eta<=0.0) mparams.eta=0.5;
               if(mparams.eta>=1.0) mparams.eta=1.0;
               ++parametersit1;
               if(parametersit1==parametersitm)break;
            }
            if(*parametersit1=="gamma1")
            {
                ++parametersit1;
                try{mparams.gamma1=stod(*parametersit1);}
                catch(...){mparams.gamma1=2.0;}
                if(mparams.gamma1<=1.0) mparams.gamma1=2.0;
                ++parametersit1;
                if(parametersit1==parametersitm)break;
            }
            if(*parametersit1=="gamma2")
            {
                ++parametersit1;
                try{mparams.gamma2=stod(*parametersit1);}
                catch(...){mparams.gamma2=0.1;}
                if(mparams.gamma2<=0.0) mparams.gamma2=0.1;
                if(mparams.gamma2>=1.0) mparams.gamma2=1.0;
                ++parametersit1;
                if(parametersit1==parametersitm)break;
            }
            if(*parametersit1=="kappa")
            {
                ++parametersit1;
                try{mparams.kappa=stod(*parametersit1);}
                catch(...){mparams.kappa=3.0;}
                if(mparams.kappa<=0.0) mparams.kappa=5.0;
                ++parametersit1;
                if(parametersit1==parametersitm)break;
            }
            if(*parametersit1=="maxit")
            {
                ++parametersit1;
                try{mparams.maxit=stoi(*parametersit1);}
                catch(...){mparams.maxit=100;}
                if(mparams.maxit<=0) mparams.maxit=100;
                ++parametersit1;
                if(parametersit1==parametersitm)break;
            }
            if(*parametersit1=="maxits")
            {
                ++parametersit1;
                try{mparams.maxits=stoi(*parametersit1);}
                catch(...){mparams.maxits=10;}
                if(mparams.maxits<=0) mparams.maxits=10;
                ++parametersit1;
                if(parametersit1==parametersitm)break;
            }
            if(*parametersit1=="print")
            {
                ++parametersit1;
                if(*parametersit1!="false") mparams.print=true;
                ++parametersit1;
                if(parametersit1==parametersitm)break;
            }
            if(*parametersit1=="tol")
            {
                ++parametersit1;
                try{mparams.kappa=stod(*parametersit1);}
                catch(...){mparams.tol=0.001;}
                if(mparams.tol<=0.0) mparams.tol=0.001;
                ++parametersit1;
                if(parametersit1==parametersitm)break;
            }
         }
    }
    string adapt="adapt";
    vector<vector<string>> ad;
    vector<vector<string>>::iterator adit;
    vector<string>::iterator adit1,aditm;
    ad=keyvar(adapt,controlvec);
    if(ad.size()>1)
    {
        cout<<"Extra adapt statement."<<endl;
        return 1;
    }
    adq scale;
    scale.adapt=false;
    scale.linselect.all=true;
    scale.quadselect.all=true;
    if(!ad.empty())
    {
        adit=ad.begin();
        aditm=adit->end();
        adit1=adit->begin();
        while(adit1<aditm)
        {
            ++adit1;
            if(*adit1=="false")
            {
                scale.adapt=false;
                break;
            }
            ++adit1;
        }
    }
    cout<<"end"<<endl;
    return 0;
}
