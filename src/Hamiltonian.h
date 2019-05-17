#include <algorithm>
#include <functional>
#include <math.h>
#include "tensor_type.h"
#include "ParametersEngine.h"
#include "Coordinates.h"
#include "MFParams.h"
#define PI acos(-1.0)

#ifndef Hamiltonian_class
#define Hamiltonian_class

extern "C" void   zheev_(char *,char *,int *,std::complex<double> *, int *, double *,
                         std::complex<double> *,int *, double *, int *);
//zheev_(&jobz,&uplo,&n,&(Ham_(0,0)),&lda,&(eigs_[0]),&(work[0]),&lwork,&(rwork[0]),&info);

class Hamiltonian {
public:

    Hamiltonian(Parameters& Parameters__, Coordinates&  Coordinates__, MFParams& MFParams__ )
        :Parameters_(Parameters__),Coordinates_(Coordinates__),MFParams_(MFParams__)

    {
        Initialize();
        Hoppings();
        HTBCreate();
    }


    void Initialize();    //::DONE
    void Hoppings();        //::DONE
    double GetCLEnergy();    //::DONE
    void InteractionsCreate();   //::DONE
    void Check_Hermiticity();  //::DONE
    void HTBCreate();   //::DONE
    double chemicalpotential(double muin,double Particles);    //::DONE

    double TotalDensity();   //::DONE
    double E_QM();   //::DONE

    void Diagonalize(char option);   //::DONE
    void copy_eigs(int i);  //::DONE

    int convert_jm_to_int(string jm_val);

    Parameters &Parameters_;
    Coordinates &Coordinates_;
    MFParams &MFParams_;
    int lx_, ly_, ns_, orbs_;
    Matrix<complex<double>> HTB_;
    Matrix<complex<double>> Ham_;
    Matrix<double> Tx,Ty,Tpxpy,Tpxmy;
    vector<double> eigs_,eigs_saved_,sx_,sy_,sz_;
    double DeltaXY_PNICTIDES;

    double HS_factor;

};



double Hamiltonian::chemicalpotential(double muin,double Particles){


    double mu_out;
    double n1,N;
    double dMubydN;
    double nstate = eigs_.size();
    dMubydN = 0.0005*(eigs_[nstate-1] - eigs_[0])/nstate;
    N=Particles;
    //temp=Parameters_.temp;
    mu_out = muin;
    bool converged=false;
    int final_i;



    if(1==2){
        for(int i=0;i<50000;i++){
            n1=0.0;
            for(int j=0;j<nstate;j++){
                n1+=double(1.0/( exp( (eigs_[j]-mu_out)*Parameters_.beta ) + 1.0));
            }
            //cout <<"i  "<< i << "  n1  " << n1 << "  mu  " << mu_out<< endl;
            if(abs(N-n1)<double(0.0001)){
                //cout<<abs(N-n1)<<endl;
                converged=true;
                final_i=i;
                break;
            }
            else {
                mu_out += (N-n1)*dMubydN;
                //cout<<i<<"    "<<n1<<"    "<<N-n1<<endl;

            }
        }

        if(!converged){
            cout<<"mu_not_converged, N = "<<n1<<endl;
        }
        else{
            //cout<<"mu converged, N = "<<n1<<" in "<<final_i<<" iters"<<endl;
        }

    }


    double mu1, mu2;
    double mu_temp = muin;
    //cout<<"mu_input = "<<mu_temp<<endl;
    if(1==1){
        mu1=eigs_[0];
        mu2=eigs_[nstate-1];
        for(int i=0;i<40000;i++){
            n1=0.0;
            for(int j=0;j<nstate;j++){
                n1+=double(1.0/( exp( (eigs_[j]-mu_temp)*Parameters_.beta ) + 1.0));
            }
            //cout <<"i  "<< i << "  n1  " << n1 << "  mu  " << mu_out<< endl;
            if(abs(N-n1)<double(0.0001)){
                //cout<<abs(N-n1)<<endl;
                converged=true;
                break;
            }
            else {
                if(n1 >N){
                    mu2=mu_temp;
                    mu_temp=0.5*(mu1 + mu_temp);
                }
                else{
                    mu1=mu_temp;
                    mu_temp=0.5*(mu2 + mu_temp);
                }

            }
            //cout<<"mu_temp = "<<mu_temp<<"   "<<n1<<endl;
        }

        if(!converged){
            cout<<"mu_not_converged, N = "<<n1<<endl;
        }
        else{
            //cout<<"mu converged, N = "<<n1<<endl;
        }

        mu_out = mu_temp;
    }

    return mu_out;
} // ----------


void Hamiltonian::Initialize(){


    ly_=Parameters_.ly;
    lx_=Parameters_.lx;
    ns_=Parameters_.ns;
    orbs_=3;
    Tx.resize(orbs_,orbs_);
    Ty.resize(orbs_,orbs_);
    Tpxpy.resize(orbs_,orbs_);
    Tpxmy.resize(orbs_,orbs_);


    int space=Coordinates_.no_dof_ ;

    HTB_.resize(space,space);
    Ham_.resize(space,space);
    eigs_.resize(space);
    eigs_saved_.resize(space);

} // ----------

double Hamiltonian::TotalDensity(){

    double n1=0.0;
    /*
    for(int j=0;j<eigs_.size();j++){
        n1 +=  1.0f/( exp(Parameters_.beta*(eigs_[j]-Parameters_.mus) ) + 1.0);
    }
    */
    return n1;

} // ----------



double Hamiltonian::E_QM(){

    double E=0.0;
    for(int j=0;j<eigs_.size();j++){
        //E +=  (eigs_[j]-Parameters_.mus)/( exp(Parameters_.beta*(eigs_[j]-Parameters_.mus) ) + 1.0);
        E +=  (eigs_[j])/( exp(Parameters_.beta*(eigs_[j]-Parameters_.mus) ) + 1.0);
    }
    return E;

} // ----------



double Hamiltonian::GetCLEnergy(){

    double EClassical=0.0;

    for(int site=0;site<ns_;site++) {
        EClassical += (-0.5*(Parameters_.U_onsite - 3*Parameters_.J_Hund)*
                       (MFParams_.Local_n[site]*MFParams_.Local_n[site]))
                +
                (2.0*Parameters_.J_Hund*
                 (MFParams_.Local_Sz[site]*MFParams_.Local_Sz[site])+
                 (MFParams_.Local_Sx[site]*MFParams_.Local_Sx[site])+
                 (MFParams_.Local_Sy[site]*MFParams_.Local_Sy[site])
                 )
                +
                (0.5*Parameters_.J_Hund*
                 (MFParams_.Local_Lz[site]*MFParams_.Local_Lz[site])+
                 (MFParams_.Local_Lx[site]*MFParams_.Local_Lx[site])+
                 (MFParams_.Local_Ly[site]*MFParams_.Local_Ly[site])
                 );

    }

    return EClassical;

} // ----------



void Hamiltonian::InteractionsCreate(){

    int space=Coordinates_.no_dof_;
    int row_, col_;
    int spin_row, spin_col;
    int orb_row, orb_col;




    for(int i=0;i<space;i++) {
        for(int j=0;j<space;j++) {
            Ham_(i,j)=HTB_(i,j);
        }
    }

    for(int site=0;site<ns_;site++) {

        //[
        //(U-3J)<n_{i}> -2J*(1 - 2spin*)<Sz_{i}> +
        //(4.0*Parameters_.J_Hund  - Parameters_.U_onsite*0.5)
        //] n_{up/dn,i}
        for(int _orb=0;_orb<3;_orb++){
            for(int _spin=0;_spin<2;_spin++){
                col_=Coordinates_.Nc_dof(site,_orb+3*_spin);
                row_=col_;
                Ham_(row_,col_) +=  one_complex*(
                            ((Parameters_.U_onsite - 3.0*Parameters_.J_Hund)
                             *MFParams_.Local_n[site])
                            - ((2.0*Parameters_.J_Hund)*(1.0 - 2.0*_spin)*
                               MFParams_.Local_Sz[site])
                            + (4.0*Parameters_.J_Hund  - Parameters_.U_onsite*0.5)
                            );
            }
        }


        //<Sminus_{i}> term + conj.
        for(int _orb=0;_orb<3;_orb++){
            spin_col=1; //down
            spin_row=0; //up
            col_=Coordinates_.Nc_dof(site,_orb+3*spin_col);
            row_=Coordinates_.Nc_dof(site,_orb+3*spin_row);
            Ham_(row_,col_) +=  -2.0*Parameters_.J_Hund*(
                        (one_complex*(MFParams_.Local_Sx[site]))
                        -(iota_complex*(MFParams_.Local_Sy[site]))
                        );
            Ham_(col_,row_) = conj(Ham_(row_,col_));
        }

        //<Lz_{i}> term + conj.
        for(int _spin=0;_spin<2;_spin++){
            orb_col=1;
            orb_row=0;
            col_=Coordinates_.Nc_dof(site,orb_col+3*_spin);
            row_=Coordinates_.Nc_dof(site,orb_row+3*_spin);
            Ham_(row_,col_) +=  -1.0*Parameters_.J_Hund*(
                        (iota_complex*(MFParams_.Local_Lz[site]))
                        );
            Ham_(col_,row_) = conj(Ham_(row_,col_));
        }

        //<Lx_{i}> term + conj.
        for(int _spin=0;_spin<2;_spin++){
            orb_col=2;
            orb_row=1;
            col_=Coordinates_.Nc_dof(site,orb_col+3*_spin);
            row_=Coordinates_.Nc_dof(site,orb_row+3*_spin);
            Ham_(row_,col_) +=  -1.0*Parameters_.J_Hund*(
                        (iota_complex*(MFParams_.Local_Lx[site]))
                        );
            Ham_(col_,row_) = conj(Ham_(row_,col_));
        }

        //<Ly_{i}> term + conj.
        for(int _spin=0;_spin<2;_spin++){
            orb_col=2;
            orb_row=0;
            col_=Coordinates_.Nc_dof(site,orb_col+3*_spin);
            row_=Coordinates_.Nc_dof(site,orb_row+3*_spin);
            Ham_(row_,col_) +=  -1.0*Parameters_.J_Hund*(
                        (-1.0*iota_complex*(MFParams_.Local_Ly[site]))
                        );
            Ham_(col_,row_) = conj(Ham_(row_,col_));
        }






    }


} // ----------


int Hamiltonian::convert_jm_to_int(string jm_val){

    int val;
    if(jm_val=="3by2_m3by2"){val=0;}
    if(jm_val=="3by2_3by2"){val=1;}
    if(jm_val=="3by2_m1by2"){val=2;}
    if(jm_val=="3by2_1by2"){val=3;}
    if(jm_val=="1by2_m1by2"){val=4;}
    if(jm_val=="1by2_1by2"){val=5;}
    return val;
}

void Hamiltonian::Check_Hermiticity()

{
    complex<double> temp(0,0);
    complex<double>temp2;

    for(int i=0;i<Ham_.n_row();i++) {
        for(int j=0;j<Ham_.n_row();j++) {
            if(
                    abs(Ham_(i,j) - conj(Ham_(j,i)))>0.00001
                    ) {
                cout<<Ham_(i,j)<<endl;
                cout<<conj(Ham_(j,i))<<endl;

            }
            assert(
                        abs(Ham_(i,j) - conj(Ham_(j,i)))<0.00001
                        ); //+ Ham_(i+orbs_*ns_,j) + Ham_(i,j+orbs_*ns_);
            //temp +=temp2*conj(temp2);
        }
    }

    // cout<<"Hermiticity: "<<temp<<endl;
}





void Hamiltonian::Diagonalize(char option){

    //extern "C" void   zheev_(char *,char *,int *,std::complex<double> *, int *, double *,
    //                       std::complex<double> *,int *, double *, int *);


    char jobz=option;
    char uplo='L'; //WHY ONLY 'L' WORKS?
    int n=Ham_.n_row();
    int lda=Ham_.n_col();
    vector<complex<double>> work(3);
    vector<double> rwork(3*n -2);
    int info;
    int lwork= -1;

    eigs_.resize(Ham_.n_row());
    fill(eigs_.begin(),eigs_.end(),0);
    // query:
    zheev_(&jobz,&uplo,&n,&(Ham_(0,0)),&lda,&(eigs_[0]),&(work[0]),&lwork,&(rwork[0]),&info);
    //lwork = int(real(work[0]))+1;
    lwork = int((work[0].real()));
    work.resize(lwork);
    // real work:
    zheev_(&jobz,&uplo,&n,&(Ham_(0,0)),&lda,&(eigs_[0]),&(work[0]),&lwork,&(rwork[0]),&info);
    if (info!=0) {
        std::cerr<<"info="<<info<<"\n";
        perror("diag: zheev: failed with info!=0.\n");
    }

    // Ham_.print();

    //  for(int i=0;i<eigs_.size();i++){
    //    cout<<eigs_[i]<<endl;
    //}


}


void Hamiltonian::HTBCreate(){

    /* Convention
 0==yz
 1==xz
 2==xy
 0==up
 1==dn
 index = orb + spin*3 + site*6;
yz_up(site=0),xz_up(site=0),xy_up(site=0), yz_dn(site=0),xz_dn(site=0),xy_dn(site=0)....site(n)...
*/

    double l_i;
    int mx=Parameters_.TBC_mx;
    int my=Parameters_.TBC_my;
    complex<double> phasex, phasey;
    int l,m,a,b;
    complex<double> Boundary_val;

    if(Parameters_.PBC==true){
        Boundary_val=one_complex;
    }
    else{
        Boundary_val=zero_complex;
    }

    HTB_.fill(0.0);


    if(!Parameters_.PNICTIDES_HOPPING){
        for(l=0;l<ns_;l++) {

            // * +x direction Neighbor
            if(Coordinates_.lx_>1){

                if(Coordinates_.indx(l)==(Coordinates_.lx_ -1)){
                    phasex=Boundary_val*exp(iota_complex*2.0*(1.0*mx)*PI/(1.0*Parameters_.TBC_cellsX));
                    phasey=one_complex;
                }
                else{
                    phasex=one_complex;
                    phasey=one_complex;
                }
                m = Coordinates_.neigh(l,0);
                for(int spin_=0;spin_<2;spin_++){
                    for (int orb1=0;orb1<3;orb1++){
                        b=Coordinates_.Nc_dof(l,orb1 + spin_*3);
                        for(int orb2=0;orb2<3;orb2++){
                            a=Coordinates_.Nc_dof(m,orb2 + spin_*3);
                            //value*c^{\dagger}_{a}c_{b}
                            assert (a!=b);
                            if(a!=b){
                                HTB_(a,b)=Parameters_.t2g_hopping_NN(orb2,orb1)*phasex;
                                HTB_(b,a)=conj(HTB_(a,b));
                            }
                        }
                    }
                }
            }


            // * +y direction Neighbor
            if(Coordinates_.ly_>1){

                if(Coordinates_.indy(l)==(Coordinates_.ly_ -1)){
                    phasex=one_complex;
                    phasey=Boundary_val*exp(iota_complex*2.0*(1.0*my)*PI/(1.0*Parameters_.TBC_cellsY));
                }
                else{
                    phasex=one_complex;
                    phasey=one_complex;
                }
                m = Coordinates_.neigh(l,2);

                for(int spin_=0;spin_<2;spin_++){
                    for (int orb1=0;orb1<3;orb1++){
                        b=Coordinates_.Nc_dof(l,orb1 + spin_*3);
                        for(int orb2=0;orb2<3;orb2++){
                            a=Coordinates_.Nc_dof(m,orb2 + spin_*3);
                            //value*c^{\dagger}_{a}c_{b}
                            assert (a!=b);
                            if(a!=b){
                                HTB_(a,b)=Parameters_.t2g_hopping_NN(orb2,orb1)*phasey;
                                HTB_(b,a)=conj(HTB_(a,b));
                            }
                        }
                    }
                }
            }
        }

        //t2g_crystal_field
        for(int i=0;i<ns_;i++) {
            for(int spin_=0;spin_<2;spin_++){
                for(int orb=0;orb<3;orb++) {
                    a=Coordinates_.Nc_dof(i,orb + 3*spin_);
                    HTB_(a,a)+=(Parameters_.Crystal_Field[orb]);
                }
            }
        }

    }
    else{
        assert(Parameters_.PNICTIDES_HOPPING == true);
        for(l=0;l<ns_;l++) {

            // Phase from As positions
            l_i=pow (-1.00, Coordinates_.indx(l) + Coordinates_.indy(l) );
            //  l_i=1.0;

            // * +x direction Neighbor
            if(Coordinates_.indx(l)==(Coordinates_.lx_ -1)){
                phasex=exp(iota_complex*2.0*(1.0*mx)*PI/(1.0*Parameters_.TBC_cellsX));
                phasey=one_complex;
            }
            else{
                phasex=one_complex;
                phasey=one_complex;
            }
            m = Coordinates_.neigh(l,0);
            for(int spin=0;spin<2;spin++) {
                for(int or1=0;or1<orbs_;or1++) {
                    for(int or2=0;or2<orbs_;or2++) {
                        a=Coordinates_.Nc_dof(l,or1 + spin*3);
                        b=Coordinates_.Nc_dof(m,or2 + spin*3);
                        assert (a!=b);
                        if(a!=b){
                            if ( (or1==2) ^ (or2==2) ) {
                                HTB_(a,b)=complex<double>(1.0*l_i*Tx(or1,or2),0.0)*phasex;
                                //if(Tx(or1,or2) != 0.0){
                                //cout<<or1<<"   "<<or2<<"  "<<l<<"  "<<m<<"   "<<HTB_(a,b)<<"   "<<endl;
                                // }
                            }
                            else {
                                HTB_(a,b)=complex<double>(1.0*Tx(or1,or2), 0.0)*phasex;
                                //                            if(Tx(or1,or2) != 0.0){
                                //                                cout<<"here"<<endl;
                                //                            }
                            }
                            HTB_(b,a)=conj(HTB_(a,b));
                        }
                    }
                }
            }


            // * +y direction Neighbor
            if(Coordinates_.indy(l)==(Coordinates_.ly_ -1)){
                phasex=one_complex;
                phasey=exp(iota_complex*2.0*(1.0*my)*PI/(1.0*Parameters_.TBC_cellsY));
            }
            else{
                phasex=one_complex;
                phasey=one_complex;
            }
            m = Coordinates_.neigh(l,2);
            for(int spin=0;spin<2;spin++) {
                for(int or1=0;or1<orbs_;or1++) {
                    for(int or2=0;or2<orbs_;or2++) {
                        a=Coordinates_.Nc_dof(l,or1 + spin*3);
                        b=Coordinates_.Nc_dof(m,or2 + spin*3);
                        assert (a!=b);
                        if(a!=b){
                            if ( (or1==2) ^ (or2==2) ) {
                                HTB_(a,b)=complex<double>(1.0*l_i*Ty(or1,or2),0.0)*phasey;
                            }
                            else {
                                HTB_(a,b)=complex<double>(1.0*Ty(or1,or2),0.0)*phasey;
                                //                        if(Ty(or1,or2) != 0.0){
                                //                            cout<<"here"<<endl;
                                //                        }
                            }
                            HTB_(b,a)=conj(HTB_(a,b));
                        }
                    }
                }
            }


            // * +x+y direction Neighbor
            phasex=one_complex;
            phasey=one_complex;
            if( Coordinates_.indy(l)==(Coordinates_.ly_ -1)  ){
                phasey=exp(iota_complex*2.0*(1.0*my)*PI/(1.0*Parameters_.TBC_cellsY));
            }
            if(Coordinates_.indx(l)==(Coordinates_.lx_ -1)){
                phasex=exp(iota_complex*2.0*(1.0*mx)*PI/(1.0*Parameters_.TBC_cellsX));
            }

            m = Coordinates_.neigh(l,4);
            for(int spin=0;spin<2;spin++) {
                for(int or1=0;or1<orbs_;or1++) {
                    for(int or2=0;or2<orbs_;or2++) {
                        a=Coordinates_.Nc_dof(l,or1 + spin*3);
                        b=Coordinates_.Nc_dof(m,or2 + spin*3);
                        assert (a!=b);
                        if(a!=b){
                            if ( (or1==2) ^ (or2==2) ) {
                                HTB_(a,b)=complex<double>(1.0*l_i*Tpxpy(or1,or2),0.0)*phasex*phasey;
                            }
                            else {
                                HTB_(a,b)=complex<double>(1.0*Tpxpy(or1,or2),0.0)*phasex*phasey;
                            }

                            HTB_(b,a)=conj(HTB_(a,b));
                        }
                    }
                }
            }


            // * +x-y direction Neighbor
            phasex=one_complex;
            phasey=one_complex;
            if( Coordinates_.indy(l)==0  ){
                phasey=exp(-1.0*iota_complex*2.0*(1.0*my)*PI/(1.0*Parameters_.TBC_cellsY));
            }
            if(Coordinates_.indx(l)==(Coordinates_.lx_ -1)){
                phasex=exp(iota_complex*2.0*(1.0*mx)*PI/(1.0*Parameters_.TBC_cellsX));
            }
            m = Coordinates_.neigh(l,7);

            for(int spin=0;spin<2;spin++) {
                for(int or1=0;or1<orbs_;or1++) {
                    for(int or2=0;or2<orbs_;or2++) {
                        a=Coordinates_.Nc_dof(l,or1 + spin*3);
                        b=Coordinates_.Nc_dof(m,or2 + spin*3);
                        assert (a!=b);
                        if(a!=b){
                            if ( (or1==2) ^ (or2==2) ) {
                                HTB_(a,b)=complex<double>(1.0*l_i*Tpxmy(or1,or2),0.0)*phasex*phasey;
                            }
                            else {
                                HTB_(a,b)=complex<double>(1.0*Tpxmy(or1,or2),0.0)*phasex*phasey;
                            }
                            HTB_(b,a)=conj(HTB_(a,b));
                        }
                    }
                }
            }

            // On-Site potential for orb = 2 (xy)
            for(int spin=0;spin<2;spin++) {
                a=Coordinates_.Nc_dof(l,2 + spin*3);
                HTB_(a,a)=complex<double>(1.0,0.0)*DeltaXY_PNICTIDES;
            }
        }

    }




    int site;
    for(int ix=0;ix<lx_;ix++){
        for(int iy=0;iy<ly_;iy++){
            for(int spin_=0;spin_<2;spin_++){
                for(int orb=0;orb<3;orb++) {
                    site=Coordinates_.Nc(ix,iy);
                    a=Coordinates_.Nc_dof(site,orb + 3*spin_);
                    HTB_(a,a)+=MFParams_.Disorder(ix,iy);
                }
            }
        }
    }



    //HTB_.print();
} // ----------



void Hamiltonian::Hoppings(){

    double t1,t2,t3,t4,t5,t6,t7,t8;

    //EXACT HAMILTONIAN FROM EQ-1,2,3 from PRB81, 014511 (2010) is implemented
    // Butfollowing have to be used to reproduce the bands shown in Fig-1 of same paper.

    int YZ_, XZ_, XY_;
    YZ_=0;
    XZ_=1;
    XY_=2;

    /*
    0==yz
    1==xz
    2==xy
    0==up
    1==dn
*/

    t1 = -0.02;         t2 = -0.06;
    t3 = -0.03;         t4 = 0.01;
    t5 = -0.2;          t6 = -0.3;
    t7 = 0.2;          t8 = 0.1;

    Tx(XZ_,XZ_)=-t2;
    Ty(YZ_,YZ_)=-t2;

    Ty(XZ_,XZ_)=-t1;
    Tx(YZ_,YZ_)=-t1;

    Tpxpy(XZ_,XZ_)=-t3;
    Tpxmy(XZ_,XZ_)=-t3;
    Tpxpy(YZ_,YZ_)=-t3;
    Tpxmy(YZ_,YZ_)=-t3;

    Tpxpy(XZ_,YZ_)=t4;
    Tpxmy(YZ_,XZ_)=-t4;
    Tpxmy(XZ_,YZ_)=-t4;
    Tpxpy(YZ_,XZ_)=t4;

    Tx(XY_,XY_)=t5;
    Ty(XY_,XY_)=t5;

    Tpxmy(XY_,XY_)=-t6;
    Tpxpy(XY_,XY_)=-t6;

    /*
    Tx(0,2)=t7;
    Tx(2,0)=t7;
    Ty(1,2)=t7;
    Ty(2,1)=t7;


    Tpxpy(0,2)=-t8;
    Tpxpy(2,0)=t8;
    Tpxmy(0,2)=-t8;
    Tpxmy(2,0)=t8;

    Tpxpy(1,2)=t8;
    Tpxpy(2,1)=-t8;
    Tpxmy(1,2)=-t8;
    Tpxmy(2,1)=t8;
    */

    Tx(XZ_,XY_)=-t7;
    Tx(XY_,XZ_)=-t7;
    Ty(YZ_,XY_)=-t7;
    Ty(XY_,YZ_)=-t7;


    Tpxpy(XZ_,XY_)=-t8;
    Tpxpy(XY_,XZ_)=t8;
    Tpxmy(XZ_,XY_)=-t8;
    Tpxmy(XY_,XZ_)=t8;

    Tpxpy(YZ_,XY_)=-t8;
    Tpxpy(XY_,YZ_)=t8;
    Tpxmy(YZ_,XY_)=t8;
    Tpxmy(XY_,YZ_)=-t8;


    DeltaXY_PNICTIDES=0.4;




} // ----------

void Hamiltonian::copy_eigs(int i){

    int space=2*ns_;

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


#endif
