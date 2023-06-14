import java.util.*;
class Student{
	int rollno;
	String name;
	
	Student(int rollno,String name)
	{
	this.rollno=rollno;
	this.name=name;
	}
public void displaystudent()
	{
	System.out.println("Name: "+name);
	System.out.println("Rollno: "+rollno);
	}
}		
class Sports{
	String sptname;
	
	Sports(String sptname)
	{
	this.sptname=sptname;
	}	
	void displaysports()
	{
	System.out.println("Sport: "+sptname);
	}
}
class Result{
	Sports Sp;
	Student Stu;
	String res;
	public Result(int rollno,String name,String sptname,String res)
	{
	this.Sp=new Sports(sptname);
	this.Stu=new Student(rollno,name);
	this.res=res;
	}
	
	void displayresult()
	{
	Stu.displaystudent();
	Sp.displaysports();
	System.out.println("Result: "+res);
	}
}		
class studentinterface
{

public static void main(String a[])
{
	int rn;
	String nm,sname,r;
	Scanner sc=new Scanner(System.in);
	System.out.println("Enter roll no: ");
	rn=sc.nextInt();
	System.out.println("Enter Name: ");
	nm=sc.next();
	System.out.println("Enter sports item: ");
	sname=sc.next();
	System.out.println("Enter result: ");
	r=sc.next();
	Result res=new Result(rn,nm,sname,r);
	res.displayresult();
	}
}
	




