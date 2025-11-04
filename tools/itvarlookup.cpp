//Find vartable entry in variable table based on name.
#include<armadillo>
using namespace arma;
using namespace std;
struct varlocs
{
    vector<int> forms;
    vector<int> positions;
    string varname;
};
bool itvlocsort(const vector<varlocs>::iterator & a,
    const vector<varlocs>::iterator & b)
    {return(a->varname<b->varname);};
vector<vector<varlocs>::iterator>::iterator itvarlookup(const string & name,
    vector<vector<varlocs>::iterator>&vartabitos)
{
    vector<varlocs> vaname(1);
    vaname[0].varname=name;
    vector<varlocs>::iterator vanameit;
    vanameit=vaname.begin();
    pair<vector<varlocs>::iterator,vector<varlocs>::iterator> it;
    it=equal_range(vartabitos.begin(),vartabitos.end(),vanameit,itvlocsort);
    return it.first;
}
