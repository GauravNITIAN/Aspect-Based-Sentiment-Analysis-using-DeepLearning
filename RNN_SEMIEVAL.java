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
public class RNN_SEMIEVAL {

    /*
     args[0] =  Maximum Sentence Size
     args[1] =  No of Training Sample
     args[2] =  No of Testing Sample
     args[3] =  LSTM layer Size
     args[4] =  miniBatch Size or No of Files
     args[5] =  bptt Length
     args[6] =  no of Pass or No of Epoches
     args[7] =  Input Size or Input Vector
     args[8] =  Output Size or Output Vector
     args[9] =  Learning Rate
     args[10] = no of Iteration to Run
     args[11] = Boolean to Load from Existing File say True - To Load Model Form Saved Model
     args[12] = InputFile Vector
     args[13] = Input OutputFile Vector
     args[14] = TestFile Vector
     args[15] = Test OutputFile Vector
     args[16] = Where the Model is going to Stored or From where model will be retrived from File
     */

    public static void main(String args[]) throws IOException
    {
        if(args.length<17)
        {
            System.out.println("Insufficient Number of Arguments ");
            System.out.println("Required Argument is 17 "+" Found  "+ args.length);
            System.exit(0);
        }
        int MAXSENTENCESIZE=Integer.parseInt(args[0]);
        int TRAININGSAMPLE=Integer.parseInt(args[1]);
        int TESTINGSAMPLE=Integer.parseInt(args[2]);
        int lstmLayerSize =Integer.parseInt(args[3]);                       //Number of units in each GravesLSTM layer
        int miniBatchSize = Integer.parseInt(args[4]);                        //Size of mini batch to use when  training
        int tbpttLength = Integer.parseInt(args[5]);                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        int numEpochs = Integer.parseInt(args[6]);                        //Total number of training epochs
        int INPUT_SIZE = Integer.parseInt(args[7]);
        int OUTPUT_SIZE = Integer.parseInt(args[8]);
        double learning_rate=Double.parseDouble(args[9]);
        int Iteration=Integer.parseInt(args[10]);
        boolean load_from_file = Boolean.parseBoolean(args[11]);
        String Infile = args[12];
        String outfile= args[13];
        String Testfile= args[14];
        String TestOutfile= args[15];
        String ModelStoreLocation=args[16];

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(learning_rate)
                .rmsDecay(0.95)
                .seed(1234)
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

        // ------------------------------------------- RELOAD Previous Saved Model ---------------------------------------------------------
        if(load_from_file)
        {
            // Load From File already saved Model with Weights
            File tempFiles = new File(ModelStoreLocation);
            tempFiles.createNewFile();
            net = ModelSerializer.restoreMultiLayerNetwork(tempFiles);
            net.init();
            net.setListeners(new ScoreIterationListener(1));
        }

        //----------------------------------------------------------------------------------------------------------------------------------


        INDArray input = InputFile(Infile, INPUT_SIZE, TRAININGSAMPLE, MAXSENTENCESIZE);
        // Create the Output Vector it Must be Comma separated for the corresponding Input
        INDArray output = InputFile(outfile, OUTPUT_SIZE, TRAININGSAMPLE, MAXSENTENCESIZE);
        // Make the DataSet of Input and Outpur
        DataSet ds = null;

        // Create Sample for Test data set
        INDArray testinput = TestFile(Testfile, INPUT_SIZE, TESTINGSAMPLE, MAXSENTENCESIZE);
        INDArray testoutput = TestFile(TestOutfile, OUTPUT_SIZE, TESTINGSAMPLE, MAXSENTENCESIZE);
        // Create the Data Set For Test data
        DataSet testingdataset = new DataSet(testinput, testoutput);

        for(int i=0;i<numEpochs;i++) {

            for(int iter=0;iter<Iteration;iter++) {

                // Create the Input Vector it Must be comma Seprated
                ds = new DataSet(input, output);

                // Set To Show the Score after Each Iteration
                net.setListeners(new ScoreIterationListener(1));

                // Fit The Data and Start Training
                net.fit(ds);
            }
            // Evaluation of Test data after each Epoch
            Evaluation eval = new Evaluation();
            INDArray outputs = net.output(testingdataset.getFeatureMatrix(), true);
            eval.evalTimeSeries(testoutput, outputs);
            System.out.println(eval.getConfusionMatrix());
            System.out.println(eval.stats());

        }

// --------------------------------------------------Write The Model------------------------------------------------------
        File tempFile = new File(ModelStoreLocation);
        tempFile.createNewFile();
        ModelSerializer.writeModel(net, tempFile, true);
// -----------------------------------------------------------------------------------------------------------------------

    }


    // Method To Load Input From File Comma Separated From Txt File

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


}

