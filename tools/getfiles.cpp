//Read data from one or more forms.  Missing observations are possible.
//Forms are specified by datafiles.  The data consist either of csv files with variable names
//as headers (headerflag=true).
//or of data files with separate files specified by varfiles 
//for variable names (headerflag=false).
//Results are observations, variable names, and flags for successful reading.
//Data are read in double-precision matrix format for each form.
//Possible formats are described in the Armadillo documentation.
//Except in the csv case with variable names as headers, data are read with auto_detect.
#include<armadillo>
#include<string.h>
using namespace std;
using namespace arma;
struct vn
{
     vector<string>varnames;
     mat vars;
};
vector<string>parse(const string & );
vector<vn>getfiles(const vector<string> & datafiles)
{
//fl locates delimiters, i and j are counters, and nd is the number of data files.
     bool flag, prev=false;
     vector<vn> results;
     size_t fl,fl1, i, j, nd;
     string str1,str2;
     results.resize(datafiles.size());
     vector<vn>::iterator resultsit;
     field<string>names;
     i=0;
     for(resultsit=results.begin();resultsit!=results.end();++resultsit)
     {
          fl=datafiles(i).find("|");
          if(fl==string::npos)
          {
               flag=resultsit->vars.load(csv_name(datafiles(i),
                     names,csv_opts::strict));
               if(!flag)
               {
                    cout<<datafiles(i)<<" not read successfully"<<endl;
                    results.clear();
                    return results;
               }
               if(resultsit->vars.n_cols!=names.n_elem)
               {
                    cout<<datafiles(i)<<" data and labels do not match."<<endl;
                    results.clear();
                    return results;
               }
               if(resultsit->vars.n_cols==0)
               {
                    cout<<datafiles(i)<<" has no content."<<endl;
                    results.clear();
                    return results;
               }
               resultsit->varnames.resize(names.n_elem);
               for(j=0;j<names.n_elem;j++)resultsit->varnames[j]=names(j);
          }
          else
          {
               str1=datafiles(i).substr(0,fl);
               str2=datafiles(i).substr(fl+1);
               if(str2=="")
               {
                    cout<<datafiles(i)<<" not in correct format.";
                    results.clear();
                    return results;
               }
               if(str1=="")
               {
                    if(prev)
                    {
                         cout<<"Only one file with no observations is permitted."<<endl;
                         results.clear();
                         return results;
                    }
                    prev=true;
//Parse variable names.
                    resultsit->varnames=parse(str2);
               }
               else  
               {
                    flag=names.load(str2);
                    if(!flag)
                    {
                         cout<<str2<<" not read successfully."<<endl;
                         results.clear();
                         return results;
                    }
                    if(names.n_elem==0)
                    {
                         cout<<str2<<" has no content."<<endl;
                         results.clear();
                         return results;
                    }
                    else  
                    {
                         resultsit->varnames.resize(names.n_elem);
                         for(j=0;j<names.n_elem;j++)resultsit->varnames[j]=names(j);
                         flag=resultsit->vars.load(str1);
                         if(!flag)
                         {
                              cout<<str1<<" not read successfully"<<endl;
                              results.clear();
                              return results;
                         }
                         if(resultsit->vars.n_cols==0)
                         {
                              if(prev)
                              {
                                   cout<<
                                       "Only one file with no observations is permitted."<<endl;
                                   results.clear();
                                   return results;
                             }
                             prev=true;
                         }
                         else
                         {
                              if(resultsit->vars.n_cols!=resultsit->varnames.size())
                              {
                                   cout<<datafiles(i)<<" data and labels do not match."<<endl;
                                   results.clear();
                                   return results;
                              }
                         }
                    }
               }
          }
          i++;
          names.clear();     
     }
     return results;
}