#include "tensor_type.h"
#include "ParametersEngine.h"


#ifndef Coordinates_class
#define Coordinates_class
class Coordinates { 

public:
    Coordinates(int lx, int ly)
        :lx_(lx),ly_(ly)
    {
        Numbering_lattice();
        Numbering_DOF();
    }

    /* Convention
 0==yz
 1==xz
 2==xy
 0==up
 1==dn
 index = orb + spin*3 + site*6;

yz_up(site=0),xz_up(site=0),xy_up(site=0), yz_dn(site=0),xz_dn(site=0),xy_dn(site=0)....site(n)...

*/
    enum {PX=0,MX,PY,MY,PXPY,MXPY,MXMY,PXMY};	// not needed - but keep as a "key"
    enum {jm_3by2_m3by2=0,jm_3by2_3by2,jm_3by2_m1by2,jm_3by2_1by2,jm_1by2_m1by2,jm_1by2_1by2};
    enum {yz_up=0,xz_up,xy_up,yz_dn,xz_dn,xy_dn};
    void Numbering_lattice();
    int indx(int i);
    int indy(int i);
    int Nc(int x, int y);

    void Numbering_DOF();
    int indx_dof(int i);
    int indy_dof(int i);
    int jm_state_dof(int i);
    int Nc_dof(int site, int dof);

    int NNc(int x, int y);
    int neigh(int site, int wneigh);
    int getneigh(int site,int wneigh);

    int lx_,ly_,ns_;
    Mat_1_int indx_,indy_;
    int no_dof_;
    Mat_1_int indx_dof_,indy_dof_,jm_state_dof_;
    Matrix<int> Nc_,neigh_;
    Matrix<int> Nc_dof_, neigh_dof_;
};

/*  
 * ***********
 *  Functions in Class Coordinates ------
 *  ***********
*/  

int Coordinates::indx(int i){
    if(i>ns_-1){perror("Coordinates.h:x-coordinate of lattice excede limit");}
    return indx_[i];
 // ----------
}


int Coordinates::indy(int i){
    if(i>ns_-1){perror("Coordinates.h:y-coordinate of lattice excede limit");}
    return indy_[i];
 // ----------
}

int Coordinates::Nc(int x, int y){
    if(x>lx_ || y>ly_){perror("Coordinates.h:ith-sitelabel of lattice excede limit (x>lx_ || y>ly_)");}
    return Nc_(x,y);
 // ----------
}


int Coordinates::indx_dof(int i){
    if(i>6*ns_-1){perror("Coordinates.h:x-coordinate of lattice excede limit");}
    return indx_dof_[i];
 // ----------
}


int Coordinates::indy_dof(int i){
    if(i>6*ns_-1){perror("Coordinates.h:y-coordinate of lattice excede limit");}
    return indy_dof_[i];
 // ----------
}



int Coordinates::jm_state_dof(int i){
    if(i>6*ns_-1){perror("Coordinates.h:jm_state_dof of lattice excede limit");}
    return jm_state_dof_[i];
 // ----------
}

int Coordinates::Nc_dof(int site, int dof){
    if(site>ns_ || dof>6){
        cout<<"site ="<<site<<endl;
        cout<<"dof = "<<dof<<endl;
        perror("Coordinates.h:ith-sitelabel of lattice excede limit (site>ns_ || dof>6)");}
    return Nc_dof_(site,dof);
 // ----------
}


int Coordinates::neigh(int site, int wneigh){
    if(site>ns_-1 || wneigh>7){
        cout<<site<<endl;
        cout<<wneigh<<endl;
        perror("Coordinates.h:getneigh -> ifstatement-2");}
    return neigh_(site,wneigh);
} // ----------


void Coordinates::Numbering_lattice(){

    ns_=lx_*ly_;

    indx_.clear(); 	indx_.resize(ns_);
    indy_.clear();	indy_.resize(ns_);
    Nc_.resize(lx_,ly_);
    neigh_.resize(ns_,8);

    // Site labeling
    int icount=0;
    for(int j=0;j<ly_;j++){
        for(int i=0;i<lx_;i++){
            indx_[icount]=i;
            indy_[icount]=j;
            Nc_(i,j)=icount;
            icount++;
        }}

    // Neighbors for each site
    for(int i=0;i<ns_;i++){ 	// ith site
        for(int j=0;j<8;j++) {		// jth neighbor
            neigh_(i,j)=getneigh(i,j);
        }
    }

} // ----------


void Coordinates::Numbering_DOF(){

    no_dof_=ns_*6;

    indx_dof_.clear(); 	indx_dof_.resize(no_dof_);
    indy_dof_.clear();	indy_dof_.resize(no_dof_);
    Nc_dof_.resize(ns_,6);
    neigh_dof_.resize(ns_,8); //not used

    // dof labeling
    int dofcount=0;
    for(int site=0;site<ns_;site++){
        for(int dof=0;dof<6;dof++){
            indx_dof_[dofcount]=indx_[site];
            indy_dof_[dofcount]=indy_[site];
            Nc_dof_(site,dof)=dofcount;
            dofcount++;
        }}

} // ----------


int Coordinates::getneigh(int site,int wneigh){

    //cout<<"here"<<endl;
    if(site>ns_-1 || wneigh>7){
        cout<<site<<endl;
        cout<<wneigh<<endl;
        perror("Coordinates.h:getneigh -> ifstatement-1");}
    int nx=indx(site);
    int ny=indy(site);
    int mx=0;
    int my=0;

    // Nearest Neighbours
    if(wneigh==0){ //PX
        mx=(nx+1)%(lx_);
        my=ny;
    }
    if(wneigh==1){ //MX
        mx=(nx+lx_-1)%(lx_);
        my=ny;
    }
    if(wneigh==2){ //PY
        mx=nx;
        my=(ny+1)%(ly_);
    }
    if(wneigh==3){ //MY
        mx=nx;
        my=(ny+ly_-1)%(ly_);
    }


    // Next-Nearest!
    if(wneigh==4){ //PXPY
        mx=(nx+1)%(lx_);
        my=(ny+1)%(ly_);
    }
    if(wneigh==5){ //MXPY
        mx=(nx+lx_-1)%(lx_);
        my=(ny+1)%(ly_);
    }
    if(wneigh==6){ //MXMY
        mx=(nx+lx_-1)%(lx_);
        my=(ny+ly_-1)%(ly_);
    }
    if(wneigh==7){ //PXMY
        mx=(nx+1)%(lx_);
        my=(ny+ly_-1)%(ly_);
    }


    return Nc_(mx,my); //Nc(mx,my);
} // ----------

#endif
