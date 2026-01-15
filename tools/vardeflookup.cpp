//Find vartable entry in variable definition table based on name.
#include<armadillo>
using namespace arma;
using namespace std;
struct xsel
{
    bool all;
    uvec list;
};
struct varlocs
{
    vector<int> forms;
    vector<int> positions;
    string varname;
};
struct vardef
{
    vector<string>catnames;
    vec o;
    bool obs;
    vector<vector<varlocs>::iterator> predits;
    vector<string> preds;
    vector<mat> predweights;
    char transform;
    char type;
    string vardefname;
    vector<string>varname;
    vector<vector<varlocs>::iterator> varnameit;
};
bool vdefsort(const vardef & a,const vardef & b);
vector<vardef>::iterator vardeflookup(const string & name,
    vector<vardef>&vardefs)
{
    vardef vaname;
    vaname.vardefname=name;
    pair<vector<vardef>::iterator,vector<vardef>::iterator> it;
    it=equal_range(vardefs.begin(),vardefs.end(),vaname,vdefsort);
    if(it.first==vardefs.end()) return it.first;
    if(next(it.first,1)!=it.second) return vardefs.end();
    return it.first;
}
