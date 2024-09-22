//Divide comma-separated string into string components.
#include<armadillo>
#include<string.h>
using namespace std;
using namespace arma;
vector<string>parse(const string & v)
{
     vector<string> results;
     size_t fcount, m;
     char r=',';
     string v1;
     fcount=count(v.begin(),v.end(),r);
     results.resize(fcount+1);
     vector<string>::iterator resultsit;
     v1=v;
     for(resultsit=results.begin();resultsit!=results.end();++resultsit)
     {
          m=v1.find(r);
          if(m==string::npos)
          {
               *resultsit=v1;
          }
          else
          {
               *resultsit=v1.substr(0,m);
               v1=v1.substr(m+1);
          }
     }
     return results;
}