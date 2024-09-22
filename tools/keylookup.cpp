//Find keyword and return value in control table.
#include<armadillo>
using namespace arma;
using namespace std;

string keylookup(const string & name, const vector<vector<string>> & controlvec)
{
    vector<vector<string>>::const_iterator it;
    const function <bool(const vector<string> & v)>f=[&name](const vector<string> & v)
         {return (v[0]==name);};
    it=find_if(controlvec.cbegin(),controlvec.cend(),f);
    if(it==controlvec.cend()) return "";
    return (*it)[1];
}
