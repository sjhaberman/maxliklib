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
struct vn{vector<string>varnames; mat vars;};
string keylookup(const string & , const vector<pair<string,string>> & );
vector<string>fstvs(const field<string> & );
vector<vn>getfiles(const vector<vector<pair<string,string>>> & datafiles){
    bool flag, prev=false;
    vector<vn> results(datafiles.size());
    string dataf, varf;
    vector<vn>::iterator resultsit;
    vector<vector<pair<string,string>>>::const_iterator datafilesit;
    field<string>names;
    datafilesit=datafiles.begin();
    for(resultsit=results.begin();resultsit!=results.end();++resultsit)
    {
        dataf=keylookup("data",*datafilesit);
        if(dataf==""){
            results.clear();
            return results;
        }
        varf=keylookup("variables",*datafilesit);
        if(varf==""){
            flag=resultsit->vars.load(csv_name(dataf,
                names,csv_opts::strict));
            if(!flag){
                cout<<dataf<<" not read successfully"<<endl;
                results.clear();
                return results;
            }
            if(resultsit->vars.n_cols!=names.n_elem){
                cout<<dataf<<" data and labels do not match."<<endl;
                results.clear();
                return results;
            }
            if(resultsit->vars.n_rows==0){
                cout<<dataf<<" has no content."<<endl;
                results.clear();
                return results;
            }
            resultsit->varnames=fstvs(names);
        }
        else{
            flag=resultsit->vars.load(dataf);
            if(!flag){
                cout<<dataf<<" not read successfully"<<endl;
                results.clear();
                return results;
            }
            if(resultsit->vars.n_rows==0){
                cout<<dataf<<" has no content."<<endl;
                results.clear();
                return results;
            }
            flag=names.load(varf);
            if(!flag){
                cout<<varf<<" not read successfully"<<endl;
                results.clear();
                return results;
            }
            if(resultsit->vars.n_cols!=names.n_elem){
                cout<<dataf<<" and "<<varf<<" do not match."<<endl;
                results.clear();
                return results;
            }
            resultsit->varnames=fstvs(names);
        }
        ++datafilesit;
        names.clear();
    }
    return results;
}
