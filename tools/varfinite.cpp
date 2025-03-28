//Find if variable is always finite and present.
//Variable name is assumed known to be valid.
#include<armadillo>
using namespace arma;
using namespace std;
struct varlocs
{
    vector<string>catnames;
    vector<int> forms;
    vec offset;
    vector<int> positions;
    vector<string> preds;
    char type;
    char transform;
    mat transition;
    string varname;
};
vector<varlocs>::iterator varlookup(const string & , vector<varlocs> &);
bool varfinite(const string & name, vector<varlocs> & vartab,
    vector<vector<int>> & patdata)
{
    size_t i;
    vector<varlocs>::iterator vartabit;
    vartabit=varlookup(name,vartab);
    i=distance(vartab.begin(),vartabit);
    vector<vector<int>>::iterator patdatait;
    for(patdatait=patdata.begin();patdatait!=patdata.end();++patdatait)
    {
        if((*patdatait)[i]==0)return false;
    }
    return true;
}
