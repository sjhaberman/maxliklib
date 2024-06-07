//Find keyword and return value in control table.
#include<armadillo>
using namespace arma;
using namespace std;

string keylookup(string & name, field<string> & source, string & def)
{
    int c,i,lower,n,pivot,r, upper;

    n=source.n_rows;
    lower=0;
    upper=n-1;
    while(lower<upper)
    {
         pivot=(lower+upper)/2;
         c=name.compare(source(pivot,0));
         if(c==0)return source(pivot,1);
         if(c<0)
         {
              upper=pivot-1;
              continue;
         }
         lower=pivot+1;
    }
    if(c==0)
    {
         return source(pivot,1);
    }
    else
    {
         return def;
    }
}
