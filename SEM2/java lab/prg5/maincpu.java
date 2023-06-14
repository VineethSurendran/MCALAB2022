import java.util.Scanner;
       class Cpu{
       int price;
       Scanner s=new Scanner(System.in);
       void get(){
                  System.out.println("Enter price:");
                  price=s.nextInt();
       }
       void display(){
                   System.out.println("CPU Price:" +price);
       }
       class Processor{
                       int numcores;
                       String manufact;
                       void get(){
                                  System.out.println("Enter no: of cores:");
                                  numcores=s.nextInt();
                                  System.out.println("Enter Manufacture name:");
                                  manufact=s.next();
                                  }
                       void display(){
                                      System.out.println("Enter no: of cores:"+numcores);
                                      System.out.println("Enter Manufacture name:"+manufact);
                                      }
                                      
       }
       static class Ram{
                    int memory;
                    String manufact;
                    Scanner s=new Scanner(System.in);
                    void get(){
                               System.out.println("Enter Memory size:");
                               memory=s.nextInt();
                               System.out.println("Enter Ram Manufacture:");
                               manufact=s.next();
                               }
                     void display(){
                                    System.out.println("Enter Memory size:"+memory); 
                                    System.out.println("Enter Ram Manufacture:"+manufact);
                                 }
                   }
}
class maincpu{
              public static void main(String args[]){
              Cpu c=new Cpu();
              Cpu.Processor p= c.new Processor();
              Cpu.Ram r=new Cpu.Ram();
              c.get();
              p.get();
              r.get();
              c.display();
              p.display();
              r.display();
              }
      }
 
              
                                      
