//Sorting vardefs by vardefname.
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
    string vardefname;;
    vector<string>varname;
    vector<vector<varlocs>::iterator> varnameit;
};
bool vdefsort(const vardef & a,const vardef & b)
    {return(a.vardefname<b.vardefname);}
