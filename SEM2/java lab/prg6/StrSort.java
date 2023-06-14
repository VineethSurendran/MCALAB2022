import java.util.*;
class Sort
{
	int n;
	String s[];
	Sort(int n)
	{
	  this.n=n;
	  s=new String[n];
	  Scanner sc=new Scanner(System.in);
	  for(int i=0;i<n;i++)
	  {
	         System.out.print("Text"+(i+1)+":");
	         s[i]=sc.nextLine();
	  }
	}
	void Sort()
	{
	  String t=" ";
	  for(int i=0;i<n;i++)
	  for(int j=0;j<n-i-1;j++)
	  if(s[j].compareTo(s[j+1])>0)
	  {
	    t=s[j];
	    s[j]=s[j+1];
	    s[j+1]=t;
	  }
	}
	void show()
	{
	  for(int i=0;i<n;i++)
	  System.out.println(s[i]);
	}
}
class StrSort{
public static void main(String[]a)
{
  Sort s=new Sort(5);
  s.Sort();
  s.show();
}
}
	
