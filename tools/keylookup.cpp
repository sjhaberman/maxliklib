//Find keyword and return value in control table.
#include<armadillo>
using namespace arma;
using namespace std;
bool vssort(const pair<string,string> & a,const pair<string,string> & b)
   {return(a.first<b.first);};
string keylookup(const string & name, const vector<pair<string,string>> & vect)
{
    pair<vector<pair<string,string>>::const_iterator,
        vector<pair<string,string>>::const_iterator> it;
    pair<string,string>vname;
    vname.first=name;
    vname.second=" ";
    it=equal_range(vect.cbegin(),vect.cend(),vname,vssort);
    if(it.first==vect.cend()) return "";
    if(next(it.first,1)!=it.second) return "";
    return (it.first->second);
}
