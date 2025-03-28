//Read data from one or more forms.  Missing observations are possible.
//Forms are specified by datafiles.  The data consist either of csv files
//with variable names as headers (headerflag=true)
//or of data files with separate files specified by varfiles 
//for variable names (headerflag=false).
//Results are observations, variable names, and flags for successful reading.
//Data are read in double-precision matrix format for each form.
//Possible formats are described in the Armadillo documentation.
//Except in the csv case with variable names as headers,
//data are read with auto_detect.
#include<armadillo>
#include<string.h>
using namespace std;
using namespace arma;
struct vn
{
    vector<string>varnames;
    mat vars;
};
vector<string>parse(const string & , const char & );
vector<vn>getfiles(const vector<vector<string>> & datafiles)
{
//fl locates delimiters, i and j are counters, and nd is the number of data files.
    bool flag, prev=false;
    vector<vn> results(datafiles.size());
    int j;
    vector<vn>::iterator resultsit;
    vector<vector<string>>::const_iterator datafilesit;
    vector<string>::const_iterator datafilesit1,datafilesit2;
    field<string>names;
    datafilesit=datafiles.begin();
    for(resultsit=results.begin();resultsit!=results.end();++resultsit)
    {
        datafilesit1=++datafilesit->begin();
        if(datafilesit->size()==2)
        {
            flag=resultsit->vars.load(csv_name(*datafilesit1,
                names,csv_opts::strict));
            if(!flag)
            {
                cout<<*datafilesit1<<" not read successfully"<<endl;
                results.clear();
                return results;
            }
            if(resultsit->vars.n_cols!=names.n_elem)
            {
                cout<<*datafilesit1<<" data and labels do not match."<<endl;
                results.clear();
                return results;
            }
            if(resultsit->vars.n_cols==0)
            {
                cout<<*datafilesit1<<" has no content."<<endl;
                results.clear();
                return results;
            }
            resultsit->varnames.resize(names.n_elem);
            for(j=0;j<names.n_elem;j++)resultsit->varnames[j]=names(j);
        }
        else
        {
            flag=resultsit->vars.load(*datafilesit1);
            if(!flag)
            {
                cout<<*datafilesit1<<" not read successfully"<<endl;
                results.clear();
                return results;
            }
            datafilesit2=next(datafilesit->begin(),3);
            flag=names.load(*datafilesit2);
            if(!flag)
            {
                cout<<*datafilesit2<<" not read successfully"<<endl;
                results.clear();
                return results;
            }
            if(resultsit->vars.n_cols!=names.n_elem)
            {
                cout<<*datafilesit1<<" and "<<*datafilesit2<<" do not match."<<endl;
                results.clear();
                return results;
            }
            resultsit->varnames.resize(names.n_elem);
            for(j=0;j<names.n_elem;j++)resultsit->varnames[j]=names(j);
        }
        ++datafilesit;
        names.clear();
    }
    return results;
}
