#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <complex>
#include <vector>
#include <math.h>
using namespace std;
#define PI_ 3.14159265

typedef vector< complex<double> >  Mat_1_Complex_doub;
typedef vector<Mat_1_Complex_doub> Mat_2_Complex_doub;

typedef vector<double>  Mat_1_doub;
typedef vector<Mat_1_doub> Mat_2_doub;

int main (){


int Length=16;

Mat_2_doub Sz, Sx, Sy;

Sz.resize(Length);Sx.resize(Length);Sy.resize(Length);
for(int i=0;i<Length;i++){
Sz[i].resize(Length);
Sx[i].resize(Length);
Sy[i].resize(Length);
}
//Runs/J_1.15/Dis_0.1/unbiased_OP_seed_101/Disorder_seed_2/output_Local_Sz.txt


double Disorder[13]= {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0, 1.2 , 1.4};
int dis_size=13;

//double Disorder[1]= {0.1};
//int dis_size=1;

char dis_char[50];


string fl_SpiSpi_out = "J_1.15_Spipi_Dis_evolution.txt";
ofstream file_SpiSpi_out(fl_SpiSpi_out.c_str());

file_SpiSpi_out<<"# Disorder_value       S_{pi,pi}"<<endl;

for (int dis_i=0;dis_i<dis_size;dis_i++){

if(Disorder[dis_i]==0.65){
sprintf(dis_char,"%.2f",Disorder[dis_i]);
}
else{
sprintf(dis_char,"%.1f",Disorder[dis_i]);
}



double S_PiPi=0;

int N_confgs=8;
for(int dis_conf=1;dis_conf<=N_confgs;dis_conf++){


char dis_conf_char[50];
sprintf(dis_conf_char,"%d",dis_conf);

string fl_in_Sz = "Runs/J_1.15/Dis_"+string(dis_char)+"/unbiased_OP_seed_101/Disorder_seed_"+string(dis_conf_char)+"/output_Local_Sz.txt";
cout<<fl_in_Sz<<endl;

string fl_in_Sx = "Runs/J_1.15/Dis_"+string(dis_char)+"/unbiased_OP_seed_101/Disorder_seed_"+string(dis_conf_char)+"/output_Local_Sx.txt";
cout<<fl_in_Sx<<endl;

string fl_in_Sy = "Runs/J_1.15/Dis_"+string(dis_char)+"/unbiased_OP_seed_101/Disorder_seed_"+string(dis_conf_char)+"/output_Local_Sy.txt";
cout<<fl_in_Sy<<endl;


ifstream file_in_Sz(fl_in_Sz.c_str());
ifstream file_in_Sx(fl_in_Sx.c_str());
ifstream file_in_Sy(fl_in_Sy.c_str());

string temp1,temp2,temp3; 
string temp;
int x,y;
double val;

//READING Sz
file_in_Sz>>temp1>>temp2>>temp3;
//cout<<temp1<<"  "<<temp2<<" "<<temp3<<endl;
for (int ix=0;ix<Length;ix++){
for (int iy=0;iy<Length;iy++){
file_in_Sz>>x>>y>>val;
//cout<<ix<<"\t"<<iy<<"\t"<<x<<"\t"<<y<<"\t"<<val<<endl;
Sz[ix][iy]=val;
}
//file_in_Sz>>temp;
//cout<<endl;
}


//READING Sx
file_in_Sx>>temp1>>temp2>>temp3;
//cout<<temp1<<"  "<<temp2<<" "<<temp3<<endl;
for (int ix=0;ix<Length;ix++){
for (int iy=0;iy<Length;iy++){
file_in_Sx>>x>>y>>val;
//cout<<ix<<"\t"<<iy<<"\t"<<x<<"\t"<<y<<"\t"<<val<<endl;
Sx[ix][iy]=val;
}
//file_in_Sz>>temp;
//cout<<endl;
}


//READING Sy
file_in_Sy>>temp1>>temp2>>temp3;
//cout<<temp1<<"  "<<temp2<<" "<<temp3<<endl;
for (int ix=0;ix<Length;ix++){
for (int iy=0;iy<Length;iy++){
file_in_Sy>>x>>y>>val;
//cout<<ix<<"\t"<<iy<<"\t"<<x<<"\t"<<y<<"\t"<<val<<endl;
Sy[ix][iy]=val;
}
//file_in_Sz>>temp;
//cout<<endl;
}


double S_PiPi_temp=0.0;

for(int ix=0;ix<Length;ix++){
for(int iy=0;iy<Length;iy++){
for(int ixp=0;ixp<Length;ixp++){
for(int iyp=0;iyp<Length;iyp++){
S_PiPi_temp += (1.0/(Length*Length*Length*Length))*(cos(1.0*(ix-ixp)*PI_))*(cos(1.0*(iy-iyp)*PI_))*
		( (Sz[ix][iy]*Sz[ixp][iyp]) + 
		  (Sx[ix][iy]*Sx[ixp][iyp]) +
		  (Sy[ix][iy]*Sy[ixp][iyp])
		);

}}}}


S_PiPi +=S_PiPi_temp*(1.0/(1.0*N_confgs));
}


file_SpiSpi_out<<dis_char<<"\t"<<S_PiPi<<endl;

}







return 0;
}
