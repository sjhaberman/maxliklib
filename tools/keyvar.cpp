//Find variables satisfying a key word.
#include<armadillo>
using namespace arma;
using namespace std;
bool svecsort(const pair<string,vector<pair<string,string>>> & a,
    const pair<string,vector<pair<string,string>>> & b)
    {return(a.first<b.first);};
vector<vector<pair<string,string>>> keyvar(const string & name, const vector<pair<string,vector<pair<string,string>>>> & controlvec ){
    vector<vector<pair<string,string>>> result;
    pair<string,vector<pair<string,string>>>t;
    t.first=name;
    vector<vector<pair<string,string>>>::iterator resultit;
    pair<vector<pair<string,vector<pair<string,string>>>>::const_iterator,
        vector<pair<string,vector<pair<string,string>>>>::const_iterator> cit;
    vector<pair<string,vector<pair<string,string>>>>::const_iterator cit1;
    cit=equal_range(controlvec.cbegin(),controlvec.cend(),t,svecsort);
    if(cit.first==controlvec.cend())return result;
    if(cit.first==cit.second)return result;
    result.resize(distance(cit.first,cit.second));
    cit1=cit.first;
    for(resultit=result.begin();resultit!=result.end();++resultit){
        resultit->resize(cit1->second.size());
        copy(cit1->second.begin(),cit1->second.end(),resultit->begin());
        ++cit1;
    }
    return result;
}

