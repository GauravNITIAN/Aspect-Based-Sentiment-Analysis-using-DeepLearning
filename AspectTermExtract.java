package com.IITPatna.Project;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;

/**
 * Created by GAURAV KUMAR on 7/18/2016.
 */
public class AspectTermExtract {

    public static void main(String args[]) throws IOException {

  AspectTermExtract as=new AspectTermExtract();
        System.out.println(as.AspectTerm(args[0]));

    }
    public static String AspectTerm(String line) throws IOException {
        line=MainCall(line);
        line=line.replace("\t","_");
        line=line.replace("\n"," ");
      return line;
    }


    public static String MainCall(String inputString) throws IOException {
        // ------------------------------------------- RELOAD Previous Saved Model ---------------------------------------------------------

        File tempFiles = new File("C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\rnnTrain.txt");
        MultiLayerNetwork net;
        net = ModelSerializer.restoreMultiLayerNetwork(tempFiles);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        //----------------------------------------------------------------------------------------------------------------------------------
        //String inputString="फेसबुक का सिक्योरिटी चैकअप फीचर पॉपअप की तरह यूजर्स को दिखाइ देगा। |";

        TRAIN_RNN trainRnn=new TRAIN_RNN();
        INDArray arr=trainRnn.Prediction(inputString,"C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\ReviewVector.txt",net,300,100);


        INDArray res= net.output(arr);
        String input=res.toString();
        Word2Vector vec=new Word2Vector();
        input=vec.Preprocess(input);
        String[] inputs=input.split(",,");
        double results1[]=vec.CommaLineSpliter(inputs[0]);
        double results2[]=vec.CommaLineSpliter(inputs[1]);
        double results3[]=vec.CommaLineSpliter(inputs[2]);
        double finalresult[]=vec.maxProbability(results1, results2, results3);
        String r=AttachClass(inputString,finalresult);
        System.out.println(r);
        return r;
    }

    public static String AttachClass(String line,double finalresult[])
    {
        String str[]=line.split(" ");
        String newline="";
        String aspect="";
        for(int i=0;i<str.length;i++)
        {
            newline=newline+str[i]+"\t";
            if(finalresult[i]==0)
            {
                aspect = "O";
            }
            else if(finalresult[i]==1)
            {
                aspect = "B_ASP";
            }
            else if(finalresult[i]==2)
            {
                aspect = "I_ASP";
            }
            newline=newline+aspect+"\n";
        }
        return newline;
    }
}
