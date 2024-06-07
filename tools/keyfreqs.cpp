//Find frequencies of keys.
#include<armadillo>
using namespace arma;
using namespace std;
struct tabentry
{
      string name;
      int freq;
      int cumfreq;
};
field<tabentry> keyfreqs(field<string> & source)
{
    int i, j, m, n, r;

    n=source.n_rows;
    umat table(n,2,fill::zeros);
    field<string>keys(n);
    string lastkey;
//First key.
    j=0;
    lastkey=source(0,0);
    keys(0)=lastkey;
    table(0,0)=1;
    if(n>1)
    {
         for(i=1;i<n;i++)
         {
              if(source(i,0)==lastkey)
              {
                   table(j,0)++;
              }
              else
              {
                   j++;
                   table(j,0)=1;
                   lastkey=source(i,0);
                   keys(j)=lastkey;
              }
         }
                   
    }
    table.col(1)=cumsum(table.col(0));
    field<tabentry> result(j+1);
    for(i=0;i<=j;i++)
    {
         result(i).name=keys(i);
         result(i).freq=table(i,0);
         result(i).cumfreq=table(i,1);
    }
    return result;    
}
