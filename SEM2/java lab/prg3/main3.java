import java.util.Scanner;
class complex{
float real;
float imag;
complex(float r,float i)
{
real=r;
imag=i;
}
complex(){
real=0;
imag=0;
}
void discomplex(){
System.out.println(real+"+"+"i"+imag);
}
void sum(complex c1,complex c2){
real=c1.real+c2.real;
imag=c1.imag+c2.imag;
}
}
class main3{
public static void main(String args[])
          {
          Scanner s1=new Scanner(System.in);
          System.out.println("Enter first num");
          System.out.println("Enter real part");
          float r1=s1.nextFloat();
          System.out.println("Enter the imaginary part");
          float i1=s1.nextFloat();
          complex c1=new complex(r1,i1);
          System.out.println("Enter second num");
          System.out.println("Enter real part");
          float r2=s1.nextFloat();
          System.out.println("Enter the imaginary part");
          float i2=s1.nextFloat();
          complex c2=new complex(r2,i2);
          System.out.println("Complex Number");
          c1.discomplex();
          c2.discomplex();
          complex c3=new complex();
          c3.sum(c1,c2);
          c3.discomplex();
          }
}
          
