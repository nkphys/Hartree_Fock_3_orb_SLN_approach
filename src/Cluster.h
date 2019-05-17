#include <algorithm>
#include <functional>
#include <math.h>
namespace SF3O{

extern "C" void   zheev_(char *,char *,int *,std::complex<double> *, int *, double *,
                         std::complex<double> *,int *, double *, int *);

class Cluster {
public:

    Cluster()
        : Parameters_(Parameters), Coordinates_(Coordinates), MFParams_(MFParams),
          lx_(Parameters_.lx), ly_(Parameters_.ly), ns_(Parameters_.ns),
          orbs_(Parameters_.orbs)
    {
        Initialize();
        Hoppings();
        HTBCreate();
    }

    FieldType eigs(int i);
    FieldType eigs_saved(int i);
    complex<FieldType> Ham(int i, int j);

    void Initialize();
    void Hoppings();
    double GetCLEnergy();
    void InteractionsCreate();
    void HTBCreate();
    double chemicalpotential(double muin,double filling);
    double TotalDensity();
    double E_QM();
    void Diagonalize(char option);
    void copy_eigs(int i);


private:
    ConstVariablestype& Parameters_;
    Coordinatetype& Coordinates_;
    MFParamsType& MFParams_;
    const int lx_, ly_, ns_, orbs_;
    Matrix<complex<FieldType>> HTB_;
    Matrix<complex<FieldType>> Ham_;
    Matrix<FieldType> Tx,Ty,Tpxpy,Tpxmy;
    vector<FieldType> eigs_,eigs_saved_,sx_,sy_,sz_;

};

/*
 * ***********
 *  Functions in Class Hamiltonian ------
 *  ***********
*/
template <typename Coordinatetype,typename ConstVariablestype,typename MFParamsType, typename FieldType>
FieldType Hamiltonian<Coordinatetype, ConstVariablestype, MFParamsType, FieldType>::
eigs(int i){return eigs_[i];}
// ----------

template <typename Coordinatetype,typename ConstVariablestype,typename MFParamsType, typename FieldType>
FieldType Hamiltonian<Coordinatetype, ConstVariablestype, MFParamsType, FieldType>::
eigs_saved(int i){return eigs_saved_[i];}
// ----------

template <typename Coordinatetype,typename ConstVariablestype,typename MFParamsType, typename FieldType>
complex<FieldType> Hamiltonian<Coordinatetype, ConstVariablestype, MFParamsType, FieldType>::
Ham(int i,int j){return Ham_(i,j);}
// ----------


template <typename Coordinatetype,typename ConstVariablestype,typename MFParamsType, typename FieldType>
double Hamiltonian<Coordinatetype, ConstVariablestype, MFParamsType, FieldType>::
chemicalpotential(double muin,double filling){
    double mu_out;
    double n1,temp,N;
    double dMubydN;
    double nstate = eigs_.size();
    dMubydN = 0.08*(eigs_[nstate-1] - eigs_[0])/nstate;
    N=filling*double(eigs_.size());
    //temp=Parameters_.temp;
    mu_out = muin;
    bool leave=false;

    for(int i=0;i<5000;i++){
        n1=0.0;
        for(int j=0;j<nstate;j++){
            n1+=double(1.0/( exp( (eigs_[j]-mu_out)*Parameters_.beta ) + 1.0));
        }
        //cout <<"i  "<< i << "  n1  " << n1 << "  mu  " << mu_out<< endl;
        if(abs(N-n1)<double(0.00001)){
            //cout<<abs(N-n1)<<endl;
          break;
        }
        else {
           mu_out += (N-n1)*dMubydN;
           //cout<<i<<"    "<<n1<<"    "<<N-n1<<endl;

        }
    }

    return mu_out;
} // ----------


template <typename Coordinatetype,typename ConstVariablestype,typename MFParamsType, typename FieldType>
void Hamiltonian<Coordinatetype, ConstVariablestype, MFParamsType, FieldType>::
Initialize(){

    int space=2*orbs_*ns_;
    Tx.resize(orbs_,orbs_);
    Ty.resize(orbs_,orbs_);
    Tpxpy.resize(orbs_,orbs_);
    Tpxmy.resize(orbs_,orbs_);
    HTB_.resize(space,space);   
    Ham_.resize(space,space);
    eigs_.resize(space);
    sx_.resize(space);
    sy_.resize(space);
    sz_.resize(space);
    eigs_saved_.resize(space);

} // ----------

template <typename Coordinatetype,typename ConstVariablestype,typename MFParamsType, typename FieldType>
double Hamiltonian<Coordinatetype, ConstVariablestype, MFParamsType, FieldType>::
TotalDensity(){
    double n1=0.0;
    for(int j=0;j<eigs_.size();j++){
        n1 +=  1.0f/( exp(Parameters_.beta*(eigs_saved_[j]-Parameters_.mus) ) + 1.0f);
    }
    return n1;
} // ----------


template <typename Coordinatetype,typename ConstVariablestype,typename MFParamsType, typename FieldType>
double Hamiltonian<Coordinatetype, ConstVariablestype, MFParamsType, FieldType>::
E_QM(){
    double E=0.0;
    for(int j=0;j<eigs_.size();j++){
        E +=  eigs_saved_[j]/( exp(Parameters_.beta*(eigs_saved_[j]-Parameters_.mus) ) + 1.0f);
    }
    return E;
} // ----------


template <typename Coordinatetype,typename ConstVariablestype,typename MFParamsType, typename FieldType>
double Hamiltonian<Coordinatetype, ConstVariablestype, MFParamsType, FieldType>::
GetCLEnergy(){
    double EClassical;
    FieldType ei, ai;
    int x,y,site;

    // Spin Components
    for(int i=0;i<lx_;i++){
      for(int j=0;j<ly_;j++){
       site = Coordinates_.Nc(i,j); //+x
       ei=MFParams_.etheta(i,j);
       ai=MFParams_.ephi(i,j);
        sx_[site] = cos(ai) * sin(ei);
        sy_[site] = sin(ai) * sin(ei);
        sz_[site] = cos(ei);
      } 
    }

    // Classical Energy
    EClassical=double(0.0);

  // NN TERMS
    for(int i=0;i<ns_;i++){
     site = Coordinates_.neigh(i,0); //+x
     EClassical += Parameters_.J_NN * (  sx_[i]*sx_[site]  + sy_[i]*sy_[site] + 1.1*sz_[i]*sz_[site] );
     site = Coordinates_.neigh(i,2); //+y
     EClassical += Parameters_.J_NN * (  sx_[i]*sx_[site]  + sy_[i]*sy_[site] + 1.1*sz_[i]*sz_[site] );

  // NNN TERMS
     site = Coordinates_.neigh(i,4); //+x+y
     EClassical += Parameters_.J_NNN * (  sx_[i]*sx_[site]  + sy_[i]*sy_[site] + 1.1*sz_[i]*sz_[site] );
     site = Coordinates_.neigh(i,7); //+x-y
     EClassical += Parameters_.J_NNN * (  sx_[i]*sx_[site]  + sy_[i]*sy_[site] + 1.1*sz_[i]*sz_[site] );

    }  
        
    return EClassical;
} // ----------



template <typename Coordinatetype,typename ConstVariablestype,typename MFParamsType, typename FieldType>
void Hamiltonian<Coordinatetype, ConstVariablestype, MFParamsType, FieldType>
::InteractionsCreate(){
    int a,b;
    int space=2*orbs_*ns_;
    FieldType ei, ai,J_Hund=Parameters_.J_HUND;

    for(int i=0;i<space;i++) {
     for(int j=0;j<space;j++) {
       Ham_(i,j)=HTB_(i,j);
     }
    }
    //HUND COUPLING
    for(int i=0;i<ns_;i++) {  // For each site
           ei=MFParams_.etheta(Coordinates_.indx(i),Coordinates_.indy(i));
           ai=MFParams_.ephi(Coordinates_.indx(i),Coordinates_.indy(i));
      for(int k=0;k<orbs_;k++) {  // For each orb
           Ham_(i+k*ns_,i+k*ns_) -=  J_Hund*( cos(ei));
           Ham_(i+k*ns_+orbs_*ns_,i+k*ns_+orbs_*ns_) -=  J_Hund*(-cos(ei));
           Ham_(i+k*ns_,i+k*ns_+orbs_*ns_) -=   J_Hund* sin(ei)*complex<double>( cos(ai),-sin(ai) ) ; //S-
           Ham_(i+k*ns_+orbs_*ns_,i+k*ns_) -=  J_Hund* sin(ei)*complex<double>( cos(ai), sin(ai) );  //S+
      }
    }

} // ----------


template <typename Coordinatetype,typename ConstVariablestype,typename MFParamsType, typename FieldType>
void Hamiltonian<Coordinatetype, ConstVariablestype, MFParamsType, FieldType>::
Diagonalize(char option){
    char jobz=option;
    char uplo='U';
    int n=Ham_.n_row();
    int lda=Ham_.n_col();
    vector<complex<FieldType>> work(3);
    vector<FieldType> rwork(3*n);
    int info,lwork= -1;


    fill(eigs_.begin(),eigs_.end(),0);
    // query:
    zheev_(&jobz,&uplo,&n,&(Ham_(0,0)),&lda,&(eigs_[0]),&(work[0]),&lwork,&(rwork[0]),&info);
    lwork = int(real(work[0]))+1;
    work.resize(lwork+1);
    // real work:
    zheev_(&jobz,&uplo,&n,&(Ham_(0,0)),&lda,&(eigs_[0]),&(work[0]),&lwork,&(rwork[0]),&info);
    if (info!=0) {
        std::cerr<<"info="<<info<<"\n";
        perror("diag: zheev: failed with info!=0.\n");
    }


}

template <typename Coordinatetype,typename ConstVariablestype,typename MFParamsType, typename FieldType>
void Hamiltonian<Coordinatetype, ConstVariablestype, MFParamsType, FieldType>
::HTBCreate(){

    int space=2*orbs_*ns_;
    int l,m,a,b;
    FieldType l_i, DeltaXY=0.4;
    HTB_.fill(0.0f);


    for(l=0;l<ns_;l++) {

        // Phase from As positions
        l_i=pow (-1.00, Coordinates_.indx(l) + Coordinates_.indy(l) );

        // * +x direction Neighbor
        m = Coordinates_.neigh(l,0);
        for(int spin=0;spin<2;spin++) {
            for(int or1=0;or1<orbs_;or1++) {
                for(int or2=0;or2<orbs_;or2++) {
                    a = l + or1*ns_ + ns_*orbs_*spin;
                    b = m + or2*ns_ + ns_*orbs_*spin;
                    if ( (or1==2) ^ (or2==2) ) {
                        HTB_(a,b)=-1.0*l_i*Tx(or1,or2);
                    }
                    else {
                        HTB_(a,b)=complex<FieldType>(-1.0*Tx(or1,or2), 0.0f);
                    }
                    HTB_(b,a)=conj(HTB_(a,b));
                }
            }
        }


        // * +y direction Neighbor
        m = Coordinates_.neigh(l,2);
        for(int spin=0;spin<2;spin++) {
            for(int or1=0;or1<orbs_;or1++) {
                for(int or2=0;or2<orbs_;or2++) {
                    a = l + or1*ns_ + ns_*orbs_*spin;
                    b = m + or2*ns_ + ns_*orbs_*spin;
                    if ( (or1==2) ^ (or2==2) ) {
                        HTB_(a,b)=complex<FieldType>(-1.0*l_i*Ty(or1,or2),0.0f);
                    }
                    else {
                        HTB_(a,b)=complex<FieldType>(-1.0*Ty(or1,or2),0.0f);
                    }
                    HTB_(b,a)=conj(HTB_(a,b));
                }
            }
        }


        // * +x+y direction Neighbor
        m = Coordinates_.neigh(l,4);
        for(int spin=0;spin<2;spin++) {
            for(int or1=0;or1<orbs_;or1++) {
                for(int or2=0;or2<orbs_;or2++) {
                    a = l + or1*ns_ + ns_*orbs_*spin;
                    b = m + or2*ns_ + ns_*orbs_*spin;
                    if ( (or1==2) ^ (or2==2) ) {
                        HTB_(a,b)=complex<FieldType>(-1.0*l_i*Tpxpy(or1,or2),0.0f);
                    }
                    else {
                        HTB_(a,b)=complex<FieldType>(-1.0*Tpxpy(or1,or2),0.0f);
                    }

                    HTB_(b,a)=conj(HTB_(a,b));
                }
            }
        }


        // * +x-y direction Neighbor
        m = Coordinates_.neigh(l,7);
        for(int spin=0;spin<2;spin++) {
            for(int or1=0;or1<orbs_;or1++) {
                for(int or2=0;or2<orbs_;or2++) {
                    a = l + or1*ns_ + ns_*orbs_*spin;
                    b = m + or2*ns_ + ns_*orbs_*spin;
                    if ( (or1==2) ^ (or2==2) ) {
                       HTB_(a,b)=complex<FieldType>(-1.0*l_i*Tpxmy(or1,or2),0.0f);
                    }
                    else {
                       HTB_(a,b)=complex<FieldType>(-1.0*Tpxmy(or1,or2),0.0f);
                    }
                       HTB_(b,a)=conj(HTB_(a,b));
                }
            }
        }

        // On-Site potential for orb = 2 (xy)
        for(int spin=0;spin<2;spin++) {
            a = l + 2*ns_ + ns_*orbs_*spin;
            HTB_(a,a)=DeltaXY;
        }
    }



} // ----------

template <typename Coordinatetype,typename ConstVariablestype,typename MFParamsType, typename FieldType>
void Hamiltonian<Coordinatetype, ConstVariablestype, MFParamsType, FieldType>
::Hoppings(){

    FieldType t1,t2,t3,t4,t5,t6,t7,t8;

    t1 = 0.02;  t2 = 0.06;
    t3 = 0.03;  t4 = -0.01;
    t5 = 0.2;   t6 = 0.3;
    t7 = -0.2;  t8 = t7/2;

    Tx(0,0)=-t2;
    Ty(1,1)=-t2;

    Ty(0,0)=-t1;
    Tx(1,1)=-t1;

    Tpxpy(0,0)=-t3;
    Tpxmy(0,0)=-t3;
    Tpxpy(1,1)=-t3;
    Tpxmy(1,1)=-t3;

    Tpxpy(0,1)=t4;
    Tpxmy(1,0)=-t4;
    Tpxmy(0,1)=-t4;
    Tpxpy(1,0)=t4;

    Tx(2,2)=t5;
    Ty(2,2)=t5;

    Tpxmy(2,2)=-t6;
    Tpxpy(2,2)=-t6;

    Tx(0,2)=-t7;
    Tx(2,0)=-t7;
    Ty(1,2)=-t7;
    Ty(2,1)=-t7;

    Tpxpy(0,2)=-t8;
    Tpxpy(1,2)=-t8;
    Tpxpy(2,0)=t8;
    Tpxpy(2,1)=t8;

    Tpxmy(0,2)=-t8;
    Tpxmy(2,0)=t8;
    Tpxmy(1,2)=t8;
    Tpxmy(2,1)=-t8;


} // ----------

template <typename Coordinatetype,typename ConstVariablestype,typename MFParamsType, typename FieldType>
void Hamiltonian<Coordinatetype, ConstVariablestype, MFParamsType, FieldType>::
copy_eigs(int i){

  int space=2*orbs_*ns_;

  if (i == 0) {
   for(int j=0;j<space;j++) {
     eigs_[j] = eigs_saved_[j];
   }  
  }
  else {
   for(int j=0;j<space;j++) {
     eigs_saved_[j] = eigs_[j];
   }  
  }

}

} // namespace


