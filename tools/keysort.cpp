//Sort control table by keys and keywords.
#include<armadillo>
using namespace arma;
using namespace std;
vector<string>parse(const string & , const char & );
vector<pair<string,vector<pair<string,string>>>> keysort(const field<string> & source){
    uword i, i1, j, n, r;
    n=source.n_rows;
    vector<pair<string,vector<pair<string,string>>>>result(n);
    vector<string>pieces;
    vector<string>pr;
    for(i=0;i<n;i++){
         pieces=parse(source(i,1),';');
         sort(pieces.begin(),pieces.end());
         r=pieces.size();
         result[i].second.resize(r);
         result[i].first=source(i,0);
         for(j=0;j<r;j++){
             pr=parse(pieces[j],'=');
             if(pr.size()!=2){
                 cout<<"Bad control file entry:"<<endl;
                 cout<<source(i,0)<<" "<<source(i,1)<<endl;
                 result.clear();
                 return result;
             }
             result[i].second[j].first=pr[0];
             result[i].second[j].second=pr[1];
             if(j>0&&result[i].second[j].first==result[i].second[j-1].first){
                 cout<<"Duplication of control file keyword:"<<endl;
                 cout<<source(i,0)<<" "<<source(i,1)<<endl;
                 result.clear();
                 return result;
             }
         }
    }
    sort(result.begin(),result.end());
    return result;
}
