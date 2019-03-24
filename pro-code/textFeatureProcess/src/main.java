import java.io.*;
import java.util.*;
public class main {
    public  static int sum = 0;
    public  static  void  main(String[] args){
        String root = "E:\\graduate-design\\";
        String name = "git-outshao" ;//"git-project\storm\storm-core\src\jvm"
        String projectPath = root + name;
        String outFile = root+"gittest27shao\\";
        File file = new File(projectPath);//File类型可以是文件也可以是文件夹
        File[] fileList = file.listFiles();//将该目录下的所有文件放置在一个File类型的数组中
        for (int i = 0; i < fileList.length; i++) {
            System.out.println(i+".....fileStart......");
            if (fileList[i].isFile()) {//判断是否为文件
                readFile(fileList[i],outFile,i);
            }
            System.out.println(".....fileEnd......");
        }
        System.out.println(main.sum);
    }
    public static void readFile(File file,String outFile,Integer index) {
        // 绝对路径或相对路径都可以，写入文件时演示相对路径,读取以上路径的input.txt文件
        //防止文件建立或读取失败，用catch捕捉错误并打印，也可以throw;
        //不关闭文件会导致资源的泄露，读写文件都同理
        //Java7的try-with-resources可以优雅关闭文件，异常时自动关闭文件；详细解读https://stackoverflow.com/a/12665271
        try (FileReader reader = new FileReader(file.getPath());
             BufferedReader br = new BufferedReader(reader) // 建立一个对象，它把文件内容转成计算机能读懂的语言
        ) {
            String line;
            //网友推荐更加简洁的写法
            NumericTexualFeature NumericTexualFeature=new NumericTexualFeature();
            List<String> texualCollection=new ArrayList<String>();
            while ((line = br.readLine()) != null) {
                // 一次读入一行数据
               String temp="";
                String[] lineArr = line.split(",");
                Integer flag = 1;
                if(lineArr.length!=30){
                    System.out.println(line);
                    System.out.println("第"+index+"个");
                }
                if(flag == 0){
                    for(int i=0;i<lineArr.length;i++){
                        if(i<19){
                            temp+=lineArr[i] + ",";
                        }else if(i<29){
                            if(lineArr[i].isEmpty()){
                                temp += " "+",";
                            }else{
                                temp += NumericTexualFeature.CamelNameSegmentation(lineArr[i]).trim() + ",";
                            }

                        }else {
                            temp+=lineArr[i] ;
                        }
                    }
                }else{
                    for(int i=0;i<lineArr.length;i++){
                        if(i<19){
                            temp+=lineArr[i] + ",";
                        }else if(i==19||(22<i&&i<29)){
                            if(lineArr[i].isEmpty()){
                                temp += " "+",";
                            }else{
                                temp += NumericTexualFeature.CamelNameSegmentation(lineArr[i]).trim() + ",";
                            }

                        }else if(i == 29){
                            temp+=lineArr[i] ;
                        }
                    }

                }


                texualCollection.add(temp);
            }
            System.out.println(texualCollection.size());
            main.sum += texualCollection.size();
            try {
               // 相对路径，如果没有则要建立一个新的output.txt文件
                if (new File(outFile+file.getName()+".csv").delete()) {
                    System.out.println("Delete existing result file.");
                }
                String fileName = file.getName();
                fileName=fileName.substring(0,fileName.lastIndexOf("."));
                File csv = new File(outFile+fileName+".csv"); // CSV数据文件
                BufferedWriter bw = new BufferedWriter(new FileWriter(csv, true)); // 附加
                for(String strList:texualCollection){
                    bw.write(strList);
                    bw.newLine();
                }
                bw.flush();
                bw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    /**
     * 写入TXT文件
     */


}
