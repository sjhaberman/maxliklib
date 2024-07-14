//Find keyword and return value in control table.
#include<armadillo>
using namespace arma;
using namespace std;

string keylookup(const string & name, const vector<vector<string>> & controlvec)
{
    int i;
    const function <bool(const vector<string> & v)>f=[&name](const vector<string> & v)
         {return (v[0]==name);};
    i=find_if(controlvec.begin(),controlvec.end(),f)-controlvec.begin();
    if(i>=controlvec.size())return "";
    return controlvec[i][1];
}
