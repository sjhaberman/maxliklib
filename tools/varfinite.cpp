//Find if variable is always finite and present.
//Variable name is assumed known to be valid.
#include<armadillo>
using namespace arma;
using namespace std;
struct varlocs{vector<uword> forms;
    vector<uword> positions; string varname;};
vector<varlocs>::const_iterator varlookup(const string & ,
    const vector<varlocs> &);
bool varfinite(const string & name, const vector<varlocs> & vartab,
    const vector<vector<uword>> & patdata){
    uword i;
    vector<varlocs>::const_iterator vartabit;
    vartabit=varlookup(name,vartab);
    i=distance(vartab.cbegin(),vartabit);
    vector<vector<uword>>::const_iterator patdatait;
    for(patdatait=patdata.cbegin();patdatait!=patdata.cend();++patdatait){
        if((*patdatait)[i]==0)return false;
    }
    return true;
}
