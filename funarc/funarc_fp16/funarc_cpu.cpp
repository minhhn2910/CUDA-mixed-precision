#include <iostream>
#include <math.h>
#include <stdio.h>

double fun( double x){
  int k, n = 5;
  double t1;
  double d1 = 1.0;
  t1 = x;
  for ( k = 1; k <= n; k++ )
    {
      d1 = 2.0 * d1;
      t1 = t1+ sin(d1 * x)/d1;
    }
    return t1;
}

float fun_float( float x){
  int k, n = 5;
  float t1;
  float d1 = 1.0;
  t1 = x;
  for ( k = 1; k <= n; k++ )
    {
      d1 = 2.0 * d1;
      t1 = t1+ sinf(d1 * x)/d1;
    }
    return t1;
}


int main( int argc, char **argv) {
  int i,n = 1000000;
  double h, t1, t2, dppi;
  double s1;
  t1 = -1.0;
  dppi = acos(t1);
  s1 = 0.0;
  t1 = 0.0;
  h = dppi / n;
  for ( i = 1; i <= n; i++)
    {
      t2 = fun(i * h);
      s1 = s1 + sqrt(h*h + (t2 - t1) * (t2 - t1));
      t1 = t2;
      if (i==100000)
        printf("%f %f \n", t2, s1);
    }

  printf("%.10f\n",s1);
  return 0;
}
