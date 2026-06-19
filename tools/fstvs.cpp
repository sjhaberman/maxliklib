//Convert string field to string vector.
#include<armadillo>
#include<string.h>
using namespace std;
using namespace arma;
vector<string>fstvs(const field<string> & v)
{
    vector<string> results;
    results.resize(v.n_elem);
    vector<string>::iterator resultsit;
    uword j=0;
    for(resultsit=results.begin();resultsit!=results.end();++resultsit){
        *resultsit=v(j);
        j++;
    }
    return results;
}
