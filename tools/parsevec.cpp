//Divide r-separated string into double-precision components.
#include<armadillo>
#include<string.h>
using namespace std;
using namespace arma;
vec parsevec(const string & v, const char & r)
{
    vector<double> results;
    size_t fcount, m;
    string v1;
    fcount=count(v.begin(),v.end(),r);
    results.resize(fcount+1);
    vector<double>::iterator resultsit;
    v1=v;
    for(resultsit=results.begin();resultsit!=results.end();++resultsit)
    {
        m=v1.find(r);
        if(m==string::npos)
        {
            try{*resultsit=stod(v1);}
            catch(...){*resultsit=datum::nan;}
        }
        else
        {
            if(m==0||m==v1.length())
            {
                cout<<"Syntax error in control record."<<endl;
                results.clear();
                return results;
            }
            try{*resultsit=stod(v1.substr(0,m));}
            catch(...){*resultsit=datum::nan;};
            v1=v1.substr(m+1);
        }
    }
    return results;
}
