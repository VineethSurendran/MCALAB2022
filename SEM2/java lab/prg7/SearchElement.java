import java.util.Scanner;
public class SearchElement{
public static void main(String[]args){
Scanner sc=new Scanner(System.in);
System.out.println("Enter the size of array");
int size=sc.nextInt();
int[]array=new int[size];
System.out.println("Enter the array elements");
for(int i=0;i<size;i++){
System.out.println("Element"+(i+1)+":");
array[i]=sc.nextInt();
}
System.out.println("Enter the element to be searched:");
int target=sc.nextInt();
boolean found=false;
for(int i=0;i<size;i++){
if(array[i]==target){
found=true;
break;
}
}
if (found){
System.out.println("Element found in the array");
}
else{
System.out.println("Element not found in the array");
}
}
}


