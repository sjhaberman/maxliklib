//Find variables satisfying a key word.
#include<armadillo>
using namespace arma;
using namespace std;
field<string> keyvar(const string & name, const vector<vector<string>> & controlvec )
{
    int h=0, i, j, k;
    const function <bool(const vector<string> & v)>f=[&name](const vector<string> & v)
         {return (v[0]==name);};
    field<string>result(0);
    
    i=find_if(controlvec.begin(),controlvec.end(),f)-controlvec.begin();
    if(i>=controlvec.size())return result;
    k=controlvec.rend()-find_if(controlvec.rbegin(),controlvec.rend(),f);
    result.set_size(k-i);
    for(j=i;j<k;j++)
    {
           result(h)=controlvec[j][1];
           h++;
    }
    return result;
}

