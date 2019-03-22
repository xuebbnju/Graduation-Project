import java.io.*;
import java.util.ArrayList;

public class NumericTexualFeature {
    static private ArrayList<String> stopWordsList=new ArrayList<String>();
    NumericTexualFeature(){
        getStopwords();
    }

//    public static void main(String[] args) {
//
//        String test= "FeatureLogThe";
//        for (String p:CamelNameSegmentation(test)) {
//            System.out.println(p);
//        }
//    }


    private static void getStopwords() {
        File stopwords=new File("resource\\stopwords.txt");
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(stopwords));
            String p=bufferedReader.readLine();
            while(p!=null){
                if(!p.equals(""))
                    stopWordsList.add(p);
                p=bufferedReader.readLine();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String CamelNameSegmentation(String str)
    {
        String temp="";
        if(str==null||str.equals(" "))return temp;
        str.trim();
        if(str!=null){
            String s = str.replaceAll("[A-Z]", " $0").toLowerCase();
            String[] s1 = s.split(" ");
            for (int i=0;i<s1.length;i++)
            {
                if(!stopWordsList.contains(s1[i])){
                    String stemming = Stemmer.stemming(s1[i]);
                    stemming.trim();
                    if(!stemming.equals(""))
                        temp += stemming + " ";
                }
            }
        }


        return temp;
    }



}

