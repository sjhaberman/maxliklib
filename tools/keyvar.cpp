//Find variables satisfying a key word.
#include<armadillo>
using namespace arma;
using namespace std;
vector<string> keyvar(const string & name, const vector<vector<string>> & controlvec )
{
    vector<string>result(0);
    vector<string>t(2),u(2);
    t[0]=name;
    t[1]=" ";
    u[0]=name;
    u[1]="~~";
    vector<string>::iterator resultit;
    vector<vector<string>>::const_iterator citb,cite;
    vector<vector<string>>::const_iterator cit;
    citb=lower_bound(controlvec.cbegin(),controlvec.cend(),t);
    cite=upper_bound(controlvec.cbegin(),controlvec.cend(),u);
    if(citb==cite)return result;
    result.resize(distance(citb,cite));
    cit=citb;
    for(resultit=result.begin();resultit!=result.end();++resultit)
    {
           *resultit=(*cit)[1];
           ++cit;
    }
    return result;
}

