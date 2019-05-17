#include <math.h>
#include "tensor_type.h"
#include "ParametersEngine.h"
#include "Coordinates.h"
#include "random"
#include <stdlib.h>
#define PI acos(-1.0)

#ifndef MFParams_class
#define MFParams_class

class MFParams{ 
public:
    /* Convention
 0==yz
 1==xz
 2==xy
 0==up
 1==dn
 index = orb + spin*3 + site*6;

yz_up(site=0),xz_up(site=0),xy_up(site=0), yz_dn(site=0),xz_dn(site=0),xy_dn(site=0)....site(n)...
*/
    // Define Fields
    Mat_1_doub Local_Sz, Local_Sx, Local_Sy;
    Mat_1_doub Local_Lz, Local_Lx, Local_Ly;
    Mat_1_doub Local_n;

    Matrix<double> Disorder;

    // Constructor
    MFParams(Parameters& Parameters__, Coordinates&  Coordinates__, mt19937_64& Generator1__ , mt19937_64& Generator2__)
        :Parameters_(Parameters__),Coordinates_(Coordinates__), Generator1_(Generator1__), Generator2_(Generator2__)
    {
        //setupm_arr();
        initialize();
    }


    double random1();
    double random2();
    void initialize();


    Parameters &Parameters_;
    Coordinates &Coordinates_;
    mt19937_64 &Generator1_; //for random fields
    mt19937_64 &Generator2_; //for random disorder
    int lx_,ly_,ns_, no_dof_;

    uniform_real_distribution<double> dis1_;//for random fields
    uniform_real_distribution<double> dis2_;//for random disorder

    //mt19937_64 mt_rand(Parameters_.RandomSeed);


};



double MFParams::random1(){

    return dis1_(Generator1_);

}

double MFParams::random2(){

    return dis2_(Generator2_);

}


void MFParams::initialize(){

    lx_=Coordinates_.lx_;
    ly_=Coordinates_.ly_;
    no_dof_=Coordinates_.no_dof_;
    ns_=Coordinates_.ns_;

    // srand(Parameters_.RandomSeed);

    Disorder.resize(lx_,ly_);

    Local_Sz.resize(ns_);
    Local_Sx.resize(ns_);
    Local_Sy.resize(ns_);

    Local_Lz.resize(ns_);
    Local_Lx.resize(ns_);
    Local_Ly.resize(ns_);

    Local_n.resize(ns_);


    ofstream Disorder_conf_file("Disorder_conf_used");
    Disorder_conf_file<<"#seed="<<Parameters_.RandomDisorderSeed<<
                        " for mt19937_64 Generator is used"<<endl;
    Disorder_conf_file<<"#ix   iy    Dis[ix,iy]"<<endl;

    ofstream Initial_OrderParams_file("Initial_OrderParams_values_generated.txt");

    if(!Parameters_.Read_OPs){
        for(int site=0;site<ns_;site++){
            Local_Sz[site] = random1();
            Local_Sx[site] = random1();
            Local_Sy[site] = random1();
            Local_Lz[site] = random1();
            Local_Lx[site] = random1();
            Local_Ly[site] = random1();
            Local_n[site] = random1();
        }
        Initial_OrderParams_file<<"#seed="<<Parameters_.RandomSeed<<
                                  " for mt19937_64 Generator is used"<<endl;
    }
    else{
        vector<string> OPstring;
        OPstring.clear();
        OPstring.push_back("All_OP");

        for(int op_no=0;op_no<OPstring.size();op_no++){
            string fl_initial_OP_in = Parameters_.File_OPs_in;
            ifstream file_initial_OP_in(fl_initial_OP_in.c_str());
            string temp1;//,temp2,temp3,temp4,temp5,temp6,temp7;
            int site_temp, x,y;
            double val_Sz, val_Sx, val_Sy, val_Lz, val_Lx, val_Ly, val_n;

            for(int i=0;i<10;i++){
                file_initial_OP_in>>temp1;
                cout<<temp1<<"   ";
            }
            cout<<endl;


            // file_initial_OP_in>>temp1>>temp2>>temp3;
            // cout<<temp1<<"  "<<temp2<<" "<<temp3<<endl;

            for(int site=0;site<ns_;site++){
                file_initial_OP_in>>site_temp>>x>>y>>val_Sz>>val_Sx>>val_Sy
                        >>val_Lz>>val_Lx>>val_Ly>>val_n;
                //cout<<ix<<"\t"<<iy<<"\t"<<x<<"\t"<<y<<"\t"<<val<<endl;
                assert(site==site_temp);
                Local_Sz[site]=val_Sz;
                Local_Sx[site]=val_Sx;
                Local_Sy[site]=val_Sy;
                Local_Lz[site]=val_Lz;
                Local_Lx[site]=val_Lx;
                Local_Ly[site]=val_Ly;
                Local_n[site]=val_n;
            }
        }

        Initial_OrderParams_file<<"#OParams are read from "<<Parameters_.File_OPs_in<<" file"<<endl;

    }



    Initial_OrderParams_file<<"#site     lx      ly     Sz[site]   Sy[site]   Sx[site]    Lz[site]   Ly[site]   Lx[site]   n[site]"<<endl;

    int site_i;
        for(int iy=0;iy<ly_;iy++){
             for(int ix=0;ix<lx_;ix++){
            site_i=Coordinates_.Nc(ix,iy);
            Initial_OrderParams_file<<site_i<<setw(15)<<ix<<setw(15)<<iy<<setw(15)<<
                                      Local_Sz[site_i]<<setw(15)<<Local_Sx[site_i]<<setw(15)<<Local_Sy[site_i]<<setw(15)
                                   <<Local_Lz[site_i]<<setw(15)<<Local_Lx[site_i]<<setw(15)<<Local_Ly[site_i]<<setw(15)
                                  <<Local_n[site_i]<<endl;
        }
        Initial_OrderParams_file<<endl;
    }



    for(int j=0;j<ly_;j++){
        for(int i=0;i<lx_;i++){
            //RANDOM Disorder
            Disorder(i,j)=Parameters_.Disorder_Strength*((2.0*random2())-1.0);
            Disorder_conf_file<<i<<"  "<<j<<"  "<<Disorder(i,j)<<endl;
        }
        Disorder_conf_file<<endl;
    }


} // ----------

#endif
