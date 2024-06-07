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
     field<string>varnames;
     mat vars;
};
field<vn>getfiles(field<string> & datafiles)
{
//i is a counter.
     bool flag;
     field<vn> results;
     int i,nd,fl,fl1;
     string str1,str2;
     nd=datafiles.n_elem;
     results.set_size(nd);
     for(i=0;i<nd;i++)
     {
          fl=datafiles(i).find("|");
          fl1=datafiles(i).length();
          if(fl<0)
          {
               flag=results(i).vars.load(csv_name(datafiles(i),
                     results(i).varnames,csv_opts::strict));
               if(!flag)
               {
                    cout<<datafiles(i)<<" not read successfully"<<endl;
                    results(i).varnames.clear();
                    return results;
               }
          }
          else
          {
               if(fl1<=fl)
               {
                    cout<<datafiles(i)<<" not in correct format.";
                    return results;
               } 
               str1=datafiles(i).substr(0,fl);
               str2=datafiles(i).substr(fl+1,fl1-fl);
               flag=results(i).vars.load(str1);
               if(!flag)
               {
                    cout<<str1<<" not read successfully"<<endl;
                    return results;
               }
               flag=results(i).varnames.load(str2);
               if(!flag)
               {
                    cout<<str2<<" not read successfully"<<endl;
                    results(i).varnames.clear();
                    return results;
               }
          }
     
     }
     return results;
}