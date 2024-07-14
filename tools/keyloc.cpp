//Find keyword and return location in control table.
#include<armadillo>
using namespace arma;
using namespace std;

int keyloc(string & name, field<string> & source)
{
    int c,i,lower,n,pivot,r, upper;

    n=source.n_rows;
    if(n==0)return -1;
    
    lower=0;
    upper=n-1;
    while(lower<=upper)
    {
         pivot=(lower+upper)/2;
         c=name.compare(source(pivot,0));
         if(c==0)return pivot;
         if(c<0)
         {
              upper=pivot-1;
              continue;
         }
         lower=pivot+1;
    }
    return -1;
}

