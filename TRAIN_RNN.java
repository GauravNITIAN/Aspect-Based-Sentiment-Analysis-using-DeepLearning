package com.IITPatna.Project;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;

/**
 * Created by GAURAV KUMAR on 7/18/2016.
 */
public class TRAIN_RNN {


    public static void main(String args[]) throws IOException {
        int MAXSENTENCESIZE=100;
        int TRAININGSAMPLE=1225;
        int TESTINGSAMPLE=1740;
        int lstmLayerSize = 210;                       //Number of units in each GravesLSTM layer
        int miniBatchSize = 32;                        //Size of mini batch to use when  training
        int exampleLength = 100;                    //Length of each training example sequence to use. This could certainly be increased
        int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        int numEpochs = 1;                        //Total number of training epochs
        int INPUT_SIZE = 300;
        int OUTPUT_SIZE = 3;
        double learning_rate=0.00001;
        int Iteration=10000;


/*
 Configuration of the Network for RNN
 */


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(0.1)
                .rmsDecay(0.95)
                .seed(12345)
                .regularization(true)
                .l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(INPUT_SIZE).nOut(lstmLayerSize)
                        .activation("tanh").build())
                .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                        .activation("tanh").build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation("softmax")        //MCXENT + softmax for classification
                        .nIn(lstmLayerSize).nOut(OUTPUT_SIZE).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                .pretrain(false).backprop(true)
                .build();


        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));



        // ------------------------------------------- RELOAD Previous Saved Model ---------------------------------------------------------
        File tempFiles = new File("C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\rnnTrain.txt");
        tempFiles.createNewFile();
        net = ModelSerializer.restoreMultiLayerNetwork(tempFiles);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        //----------------------------------------------------------------------------------------------------------------------------------

        String testingfile = "C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\Test\\Test.txt";
        String outputfile = "C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\Test\\TestOutput.txt";

        INDArray testinput = TestFile(testingfile, INPUT_SIZE, TESTINGSAMPLE, MAXSENTENCESIZE);
        INDArray testoutput = TestFile(outputfile, OUTPUT_SIZE, TESTINGSAMPLE, MAXSENTENCESIZE);
        DataSet testingdataset = new DataSet(testinput, testoutput);
        INDArray output=null;
        DataSet ds = null;

        for(int i=0;i<numEpochs;i++) {
            for (int batch = 0; batch < 1; batch++)
            {

                //    String Inputfile = "C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\Train\\Train.txt_vector.txtinput" + batch + ".txt";
                String Inputfile = "C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\NewData\\Train.txt_vector.txtinput0.txt";

                //    String Outfile = "C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\TrainOutput\\TRAINDATA.txtSentenceOutput.txtinput" + batch + ".txt";
                String Outfile = "C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\NewData\\TRAINDATA.txtSentenceOutput.txtinput0.txt";

                System.out.println(TRAININGSAMPLE);

                INDArray input = InputFile(Inputfile, INPUT_SIZE, TRAININGSAMPLE, MAXSENTENCESIZE);
                output = InputFile(Outfile, OUTPUT_SIZE, TRAININGSAMPLE, MAXSENTENCESIZE);

                ds = new DataSet(input, output);


                net.fit(ds);

            }


        }



        Evaluation eval = new Evaluation();
        Evaluation eval1 = new Evaluation();


        // Evaluation.evalTimeSeries(INDArray labels, INDArray predicted, INDArray outputMask)
        INDArray outputs = net.output(testingdataset.getFeatureMatrix(), true);
        INDArray outs = net.output(ds.getFeatureMatrix(), true);

        eval.evalTimeSeries(testoutput, outputs);
        eval1.evalTimeSeries(output,outs);
        // -----------------------------------------------------------------------------------
        BufferedWriter buffer =new BufferedWriter(new FileWriter(new File("output.txt")));
        buffer.write(outputs.toString());
        buffer.close();
        //eval.eval(testoutput,outputs);

        //   System.out.println(ds.getLabels());
        System.out.println(eval.getConfusionMatrix());
        System.out.println(eval.stats());
        System.out.println("-------------------------Training  data---------------------------");
        System.out.println(eval1.stats());


// --------------------------------------------------Write The Model------------------------------------------------------
        File tempFile = new File("C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\rnnTrain.txt");
        tempFile.createNewFile();
        ModelSerializer.writeModel(net, tempFile, true);

// -----------------------------------------------------------------------------------------------------------------------


        PrintWriter pw = new PrintWriter("layer.txt");

        for(org.deeplearning4j.nn.api.Layer layer : net.getLayers()) {
            pw.println("*************************  LAYER  ********************");
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
            pw.println(w);
            //System.out.println("Weights: " + w);
        }
        pw.close();

    }

    public static INDArray InputFile(String file,int INPUTSIZE,int TRAININGSAMPLE,int MAXSENTENCESIZE) throws FileNotFoundException, IOException {

        BufferedReader hidden = new BufferedReader(new FileReader(new File(file)));
        String line;
        INDArray inputs = Nd4j.zeros(TRAININGSAMPLE, INPUTSIZE,MAXSENTENCESIZE);


        for(int j=0;j<TRAININGSAMPLE;j++) {
            //      System.out.println("aaaaaaaaaaa"+j);

            int count = 0;
            line = hidden.readLine();
            String str1[] = line.split(",");
            for (int i = 0; i < MAXSENTENCESIZE; i++) {

                for (int c = 0; c < INPUTSIZE; c++) {
                    //Double b=Double.parseDouble(str1[count]);

                    inputs.putScalar(new int[]{j, c, i}, Double.parseDouble(str1[count]));
                    count++;
                }
            }
        }
        return inputs;
    }



    public static INDArray MakeInput(String line,int INPUTSIZE,int TRAININGSAMPLE,int MAXSENTENCESIZE) throws FileNotFoundException, IOException {


        INDArray inputs = Nd4j.zeros(TRAININGSAMPLE, INPUTSIZE,MAXSENTENCESIZE);

        for(int j=0;j<TRAININGSAMPLE;j++) {
            int count = 0;


            String str1[] = line.split(",");
            for (int i = 0; i < MAXSENTENCESIZE; i++) {

                for (int c = 0; c <INPUTSIZE ; c++) {

                    inputs.putScalar(new int[]{j, c, i}, Double.parseDouble(str1[count]));
                    count++;
                }
            }


            // input.put

        }
        return inputs;
    }

    public static INDArray TestFile(String file,int INPUTSIZE,int TESTINGSAMPLE,int MAXSENTENCESIZE) throws FileNotFoundException, IOException {

        BufferedReader hidden = new BufferedReader(new FileReader(new File(file)));
        String line;
        INDArray inputs = Nd4j.zeros(TESTINGSAMPLE, INPUTSIZE,MAXSENTENCESIZE);


        for(int j=0;j<TESTINGSAMPLE;j++) {
            int count = 0;
            line = hidden.readLine();
            String str1[] = line.split(",");
            for (int i = 0; i < MAXSENTENCESIZE; i++) {

                for (int c = 0; c < INPUTSIZE; c++) {

                    inputs.putScalar(new int[]{j, c, i}, Double.parseDouble(str1[count]));
                    count++;
                }
            }
        }
        return inputs;
    }

    public INDArray Prediction(String sentence, String file, MultiLayerNetwork net, int INPUT_SIZE, int MAXSENTENCESIZE) throws IOException {
        File f=new File(file);

        String sen[]=sentence.split(" ");
        String lineTOwrite="";
        for(int i=0;i<sen.length;i++)
        {
            BufferedReader buffer=new BufferedReader(new FileReader(f));
            String line;
            String append="";
            while((line=buffer.readLine())!=null)
            {
                String[] str=line.split(" ");

                if(str[0].equals(sen[i]))
                {
                    line=line.replace(str[0],"");
                    line=line.trim();
                    line=line.replace(" ",",");
                    append=line;
                    break;
                }
            }
            if(append=="")
            {
                lineTOwrite=lineTOwrite+","+NULLString(INPUT_SIZE);
            }
            else
            {
                lineTOwrite=lineTOwrite+","+line;
            }


            buffer.close();
        }

        String[] st=lineTOwrite.split(",");
        int no_of_more_null=(MAXSENTENCESIZE-(st.length/INPUT_SIZE));
        for(int i=0;i<no_of_more_null;i++)
        {
            lineTOwrite=lineTOwrite+","+NULLString(INPUT_SIZE);
        }
        lineTOwrite=lineTOwrite.replaceFirst(",","");
        return  MakeInput(lineTOwrite,INPUT_SIZE, 1, MAXSENTENCESIZE);

    }
    public String NULLString(int INPUT_SIZE)
    {
        String nullString ="";
        for (int i = 0; i < INPUT_SIZE; i++) {
            nullString = nullString.concat("," + String.valueOf(0.0));

        }
        nullString=nullString.trim();
        nullString=nullString.replaceFirst(",","");


        return nullString;
    }
}

