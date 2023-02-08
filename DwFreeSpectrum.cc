#include <stdio.h>
#include <math.h>

const int L=16;
const double pi = M_PI;

int main(int argc,char **argv)
{
  double m=0.0;
  for(int x=0;x<L;x++){
  for(int y=x;y<L;y++){
  for(int z=y;z<L;z++){
  for(int t=z;t<L;t++){
    double px=2*x*pi/L;
    double py=2*y*pi/L;
    double pz=2*z*pi/L;
    double pt=2*t*pi/L;

    double lambda_r = m + 4.0-cos(px)-cos(py)-cos(pz)-cos(pt);
    double lambda_i = sin(px)*sin(px)+sin(py)*sin(py)+sin(pz)*sin(pz)+sin(pt)*sin(pt);
    lambda_i = sqrt(lambda_i);

    printf("%le %le\n",lambda_r,lambda_i);
    printf("%le %le\n",lambda_r,-lambda_i);
  }
  printf("\n");  }
  printf("\n");  }
  printf("\n");  }
}
