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
    xsel constant;
    xsel deg1;
    xsel full;
    vec o;
    bool obs;
    vector<string> preds;
    vector<vector<varlocs>::iterator> predits;
    char transform;
    char type;
    string vardefname;;
    vector<string>varname;
    vector<vector<varlocs>::iterator> varnameit;
};
bool vdefsort(const vardef & a,const vardef & b);
vector<vardef>::const_iterator vardeflookup(const string & name,
    const vector<vardef>&vardefs)
{
    vardef vaname;
    vaname.vardefname=name;
    pair<vector<vardef>::const_iterator,vector<vardef>::const_iterator> it;
    it=equal_range(vardefs.cbegin(),vardefs.cend(),vaname,vdefsort);
    if(it.first==vardefs.cend()) return it.first;
    if(next(it.first,1)!=it.second) return vardefs.cend();
    return it.first;
}
