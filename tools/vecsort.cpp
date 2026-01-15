//Sorting vecs.
#include<armadillo>
using namespace arma;
using namespace std;
bool vecsort(const vec & a,const vec & b){
    vec::const_iterator ait,bit;
    if(b.is_empty())return false;
    if(a.is_empty())return true;
    bit=b.cbegin();
    for(ait=a.cbegin();ait!=a.cend();++ait){
        if(bit==b.cend())return false;
        if(*ait<*bit)return true;
        if(*ait>*bit)return false;
        ++bit;
    }
    return false;
}
 
