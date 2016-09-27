package com.IITPatna.Project;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.w3c.dom.Attr;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.*;
import java.util.*;
import java.util.regex.PatternSyntaxException;

/**
 * Created by GAURAV KUMAR on 7/18/2016.
 */
public class Word2Vector {

/*


          batchSize is the amount of words you process at a time.
          minWordFrequency is the minimum number of times a word must appear in the corpus. Here, if it appears less than 5 times, it is not learned. Words must appear in multiple contexts to learn useful features about them. In very large corpora, it’s reasonable to raise the minimum.
         useAdaGrad - Adagrad creates a different gradient for each feature. Here we are not concerned with that.
          layerSize specifies the number of features in the word vector. This is equal to the number of dimensions in the featurespace. Words represented by 500 features become points in a 500-dimensional space.
          iterations this is the number of times you allow the net to update its coefficients for one batch of the data. Too few iterations mean it may not have time to learn all it can; too many will make the net’s training longer.
         learningRate is the step size for each update of the coefficients, as words are repositioned in the feature space.
         minLearningRate is the floor on the learning rate. Learning rate decays as the number of words you train on decreases. If learning rate shrinks too much, the net’s learning is no longer efficient. This keeps the coefficients moving.
        iterate tells the net what batch of the dataset it’s training on.
        tokenizer feeds it the words from the current batch.
       vec.fit() tells the configured net to begin training.


 */

    static Map<String, String> map;
    static String[] WordToINDEX;
    static int layerSize = 300;
    static String nullString = "";
    static int MAXSentenceSize = 100;

    public static void main(String args[]) throws IOException {


        for (int i = 0; i < layerSize; i++) {
            nullString = nullString.concat("," + String.valueOf(0.0));

        }
        System.out.println(nullString);

        Word2Vector vec = new Word2Vector();
        // int Number_of_Vector_Sample=317111;
        // WordToINDEX =new String[Number_of_Vector_Sample];
        try {
            // vec.ConvertVector("C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\TrainReview.txt","C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\ReviewVector.txt");

            //  vec.WordToMap("C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\ReviewVector.txt");
            // vec.WriteVectorSentence("C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\Train.txt",MAXSentenceSize);
            //  vec.BIOFileToSentence("C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\BIOTRAIN.txt");
            // vec.parse("C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\Train.txt_vector.txt");
            //vec.BreakFile("C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\NewData\\Train.txt_vector.txtinput0.txt",10);
            // vec.PredictedResult("output.txt","TestReview.txt");
            // vec.BIOFileToSentence("result.txt");

            vec.ExtractAspectTerm("result.txt");
            vec.XMLCreate("result.txtaspect.txt");
//        for(int i=0;i<map.size();i++)
            {
                // System.out.println(map.get(WordToINDEX[i]));
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        //     vec.LoadVectors("C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\ReviewVector.txt");

    }

    public INDArray SentenceToINDArray(String sentence) {
        String[] str = sentence.split(" ");
        INDArray inputs = Nd4j.zeros(str.length, layerSize);
        for (int i = 0; i < str.length; i++) {
            String word = map.get(str[i]);
            String[] input = word.split(",");
            Double dou[] = new Double[input.length];

            for (int j = 0; j < input.length; j++) {
                dou[j] = Double.parseDouble(input[j]);
                inputs.putScalar(new int[]{i, j}, dou[j]);
            }
        }
        return inputs;
    }

    public void PredictBIO(INDArray input, MultiLayerNetwork net) {
        int[] res = net.predict(input);
        for (int i = 0; i < res.length; i++)
            System.out.println(res[i]);
    }

    public void parse(String file) throws IOException {
        File fn = new File(file);

        BufferedReader buffer = new BufferedReader(new FileReader(fn));
        BufferedWriter buffer1 = new BufferedWriter(new FileWriter(file + "INVector.txt"));
        String line;
        while ((line = buffer.readLine()) != null) {

            String str = line.trim();
            str = str.replace(" ", ",");
            buffer1.write(str);
            buffer1.newLine();
        /*    String[] str=line.split(",");
            for(int i=1;i<str.length;i++)
            {
                Double value= Double.parseDouble(str[i]);
                System.out.println(value);
            }  */
        }
        buffer1.close();
        buffer.close();
    }

    public void WordToMap(String vectorFile) throws IOException {

        // Map means Here Vocab Here
        map = new HashMap<>();
        File fn = new File(vectorFile);
        int count = 0;
        BufferedReader buffer = new BufferedReader(new FileReader(fn));
        String line;
        while ((line = buffer.readLine()) != null) {
            try {
                String[] str = line.split(" ");
                String text = str[0];
                //  WordToINDEX[count++] = text;
                String value = line.replaceFirst(str[0], "");
                value = value.replaceAll(" ", ",");
                map.put(text, value);

            } catch (PatternSyntaxException e) {
                String[] str = line.split(" ");
                String temp = "";
                for (int i = 1; i < str.length; i++) {
                    temp = temp.concat(str[i] + " ");

                }
                temp = temp.trim();
                temp = temp.replaceAll(" ", ",");
                map.put(str[0], temp);
                continue;

            }
        }

    }


    public void ConvertVector(String file, String Outfile) throws IOException {
        SentenceIterator iter = new LineSentenceIterator(new File(file));
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                //  System.out.println(sentence.toLowerCase());
                return sentence.toLowerCase();
            }
        });


        final EndingPreProcessor preProcessor = new EndingPreProcessor();
        TokenizerFactory tokenizer = new DefaultTokenizerFactory();
        tokenizer.setTokenPreProcessor(new TokenPreProcess() {
            @Override
            public String preProcess(String token) {
                token = token.toLowerCase();
                String base = preProcessor.preProcess(token);
                base = base.replaceAll("\\d", "d");
                if (base.endsWith("ly") || base.endsWith("ing"))
                    System.out.println();
                return base;
            }
        });

        int batchSize = 1000;
        int iterations = 3;


        System.out.println("Buliding Model.....");
        Word2Vec vec = new Word2Vec.Builder()
                .batchSize(batchSize) //# words per minibatch.
                .minWordFrequency(1) //
                .useAdaGrad(false) //
                .layerSize(layerSize) // word feature vector size
                .iterations(iterations) // # iterations to train
                .learningRate(0.025) //
                .minLearningRate(1e-3) // learning rate decays wrt # words. floor learning
                .negativeSample(10) // sample size 10 words
                .iterate(iter) //
                .tokenizerFactory(tokenizer)
                .build();
        vec.fit();
        try {
            WordVectorSerializer.writeWordVectors(vec, "C:\\Users\\GAURAV KUMAR\\Desktop\\DeepLearning\\ReviewVector.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public void LoadVectors(String file) throws FileNotFoundException, UnsupportedEncodingException {

        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File(file));
        WeightLookupTable weightLookupTable = wordVectors.lookupTable();
        Iterator<INDArray> vectors = weightLookupTable.vectors();
        INDArray wordVector = wordVectors.getWordVectorMatrix("myword");
        // double[] wordVector = wordVectors.getWordVector("myword");


    }

    public void BIOFileToSentence(String file) throws IOException {
        File fn = new File(file);

        BufferedReader buffer = new BufferedReader(new FileReader(fn));
        BufferedWriter out = new BufferedWriter(new FileWriter(new File(file + "SentenceReview.txt")));
        String line;
        String lineToWrite = "";
        while ((line = buffer.readLine()) != null) {
            String[] str = line.split("\t");
            if (line.equals("EOL\tO")) {
                lineToWrite = lineToWrite.trim();
                out.write(lineToWrite);
                out.newLine();
                buffer.readLine();
                lineToWrite = "";
            } else {
                lineToWrite = lineToWrite.concat(str[0] + " ");
            }
        }
        buffer.close();
        out.close();

    }

    public void WriteVectorSentence(String BIOfile, int MaxLength) throws IOException {
        File fn = new File(BIOfile);
        BufferedReader buffer = new BufferedReader(new FileReader(fn));
        BufferedWriter buffer1 = new BufferedWriter(new FileWriter(new File(BIOfile + "Sentence.txt"), true));
        BufferedWriter buffer2 = new BufferedWriter(new FileWriter(new File(BIOfile + "SentenceOutput.txt"), true));
        String line;
        ArrayList<String> ar = new ArrayList<String>();
        ArrayList<String> out = new ArrayList<String>();
        while ((line = buffer.readLine()) != null) {

            if (line.equals("EOL\tO")) {
                String lineTowrite = "";
                String OutputTowrite = "";
                for (int i = 0; i < MaxLength; i++) {

                    try {
                        if (map.get(ar.get(i)) != null) {
                            if (i == MaxLength - 1) {
                                lineTowrite = lineTowrite.concat(map.get(ar.get(i)));
                                OutputTowrite = OutputTowrite.concat(out.get(i));
                            } else {
                                lineTowrite = lineTowrite.concat(map.get(ar.get(i)) + ",");
                                OutputTowrite = OutputTowrite.concat(out.get(i) + ",");

                            }
                        } else {
                            if (i == MaxLength - 1) {
                                lineTowrite = lineTowrite.concat(nullString);
                                OutputTowrite = OutputTowrite.concat("O");
                            } else {
                                lineTowrite = lineTowrite.concat(nullString + ",");
                                OutputTowrite = OutputTowrite.concat("O" + ",");
                            }
                        }
                    } catch (Exception e) {
                        if (i == MaxLength - 1) {
                            lineTowrite = lineTowrite.concat(nullString);
                            OutputTowrite = OutputTowrite.concat("O");
                        } else {
                            lineTowrite = lineTowrite.concat(nullString + ",");
                            OutputTowrite = OutputTowrite.concat("O" + ",");
                        }
                        continue;
                    }
                }
                // Write it in the file
                lineTowrite = lineTowrite.trim();
                lineTowrite = lineTowrite.replaceFirst(",", "");
                lineTowrite = lineTowrite.replaceAll(",,", ",");
                // String str=lineTowrite.split()
                buffer1.write(lineTowrite);
                buffer1.newLine();


                OutputTowrite = OutputTowrite.trim();
                // OutputTowrite=OutputTowrite.replaceFirst(",","");
                //   OutputTowrite=OutputTowrite.replaceAll(",,",",");
                // OutputTowrite=OutputTowrite.replaceAll("/*,","");
                //   OutputTowrite=OutputTowrite.replaceAll("/?,","");
                //  OutputTowrite=OutputTowrite.replaceAll("|,","");
                // Give Them Values
                OutputTowrite = OutputTowrite.replaceAll("B_ASP", "0,1,0");
                OutputTowrite = OutputTowrite.replaceAll("I_ASP", "0,0,1");
                OutputTowrite = OutputTowrite.replaceAll("O", "1,0,0");

                buffer2.write(OutputTowrite);
                buffer2.newLine();

                ar.clear();
                out.clear();
                buffer.readLine();
            } else {
                String[] str = line.split("\t");

                try {
                    ar.add(str[0]);
                    System.out.println("aaaaaaaaaaa" + str[0] + "   " + str[1]);
                    out.add(str[1]);
                } catch (ArrayIndexOutOfBoundsException e) {
                    out.add("O");
//                     System.out.println("aaaaaaaaaaa" + str[0] + "   " + str[1]);
                    continue;

                }
            }
        }
        buffer.close();
        buffer1.close();
        buffer2.close();
    }

    public void BreakFile(String file, int NUMBER) throws IOException {

        File fn = new File(file);
        BufferedReader counter = new BufferedReader(new FileReader(fn));
        String str;
        int count = 0;
        while ((str = counter.readLine()) != null) {
            count++;
        }
        counter.close();
        int LINES = count / NUMBER;

        BufferedReader buffer = new BufferedReader(new FileReader(fn));
        BufferedWriter out;
        String line;
        String files[] = new String[NUMBER + 1];
        for (int i = 0; i <= NUMBER; i++) {
            files[i] = file + "input" + String.valueOf(i) + ".txt";
            out = new BufferedWriter(new FileWriter(files[i], true));
            int start = 0;
            while ((line = buffer.readLine()) != null) {
                line = line.trim();
                line = line.replaceAll(" ", ",");
                if (start >= (LINES - 1)) {
                    out.write(line);
                    out.newLine();
                    out.close();
                    break;
                } else {
                    start++;
                    out.write(line);
                    out.newLine();
                }
            }
            if (out != null)
                out.close();
        }
    }

    public void PredictedResult(String file1, String file2) throws IOException {
        File f1 = new File(file1);
        File f2 = new File(file2);
        File f3 = new File("result.txt");
        f3.createNewFile();
        BufferedReader buffer1 = new BufferedReader(new FileReader(f1));
        BufferedReader buffer2 = new BufferedReader(new FileReader(f2));
        BufferedWriter out = new BufferedWriter(new FileWriter(f3, true));
        String line1, line2, line3, line4;
        String sentence;
        double array1[], array2[], array3[];
        double[] prediction;
        while ((line1 = buffer1.readLine()) != null) {
            line2 = buffer1.readLine();
            line3 = buffer1.readLine();
            line4 = buffer1.readLine();

            line1 = Preprocess(line1);
            line2 = Preprocess(line2);
            line3 = Preprocess(line3);

            array1 = CommaLineSpliter(line1);
            array2 = CommaLineSpliter(line2);
            array3 = CommaLineSpliter(line3);
            prediction = maxProbability(array1, array2, array3);

            // File reading to Label the Output we have
            sentence = buffer2.readLine();
            String[] str = sentence.split(" ");


            for (int i = 0; i < str.length; i++) {
                String aspect = null;
                if (prediction[i] == 0.0)
                    aspect = "O";
                else if (prediction[i] == 1.0)
                    aspect = "B_ASP";
                else if (prediction[i] == 2.0)
                    aspect = "I_ASP";

                out.write(str[i] + "\t" + aspect);
                out.newLine();
            }
            out.write("EOL\tO");
            out.newLine();
            out.newLine();


        }
        buffer1.close();
        buffer2.close();
        out.close();
    }

    public String Preprocess(String line) {
        line = line.replace("[", "");
        line = line.replace("]", "");
        line = line.replace(",", "");
        line = line.trim();
        line = line.replace(" ", ",");

        return line;
    }

    public double[] CommaLineSpliter(String line) {
        String[] split = line.split(",");
        double value[] = new double[split.length];
        for (int i = 0; i < split.length; i++) {
            value[i] = Double.parseDouble(split[i]);
        }
        return value;
    }

    public double[] maxProbability(double array1[], double array2[], double array3[]) {
        double prediction[] = new double[array1.length];
        for (int i = 0; i < array1.length; i++) {
            if (array1[i] > array2[i]) {
                if (array1[i] > array3[i]) {
                    prediction[i] = 0;
                } else {
                    prediction[i] = 2;
                }
            } else {
                if (array2[i] > array3[i]) {
                    prediction[i] = 1;
                } else {
                    prediction[i] = 2;
                }
            }
        }
        return prediction;
    }


    public void ExtractAspectTerm(String file) throws IOException {
        File fn = new File(file);

        BufferedReader buffer = new BufferedReader(new FileReader(fn));
        BufferedWriter out = new BufferedWriter(new FileWriter(new File(file + "SentenceReview.txt")));
        String line;
        String lineToWrite = "";
        while ((line = buffer.readLine()) != null) {
            String[] str = line.split("\t");
            if (line.equals("EOL\tO")) {
                lineToWrite = lineToWrite.trim();
                out.write(lineToWrite);
                out.newLine();
                buffer.readLine();
                lineToWrite = "";
            } else {
                lineToWrite = lineToWrite.concat(str[0] + " "+str[1]+"\t");
            }
        }
        buffer.close();
        out.close();

        buffer = new BufferedReader(new FileReader(new File(file+"SentenceReview.txt")));
        out = new BufferedWriter(new FileWriter(new File(file + "aspect.txt")));
        while((line=buffer.readLine())!=null)
        {
            String lines ="";
            String[] str=line.split("\t");
            String review="";
            for(int i=0;i<str.length;i++)
            {

                String[] st=str[i].split(" ");
                review=review+" "+st[0];
                if(st[1].equals("O"))
                {

                }
                else if(st[1].equals("B_ASP"))
                {
                    lines=lines+"\t"+st[0];
                    while(true)
                    {
                        try {
                            st = str[++i].split(" ");
                            review=review+" "+st[0];
                        }
                        catch(ArrayIndexOutOfBoundsException e)
                        {
                            break;
                        }
                        if(st[1].equals("I_ASP"))
                        {
                            lines=lines+" "+st[0];
                        }
                        else if(st[1].equals("B_ASP"))
                        {
                            i--;
                            break;
                        }
                        else
                        {
                            break;
                        }
                    }
                }
                /* Line to Write; */
                lines=lines.trim();
                // lines=lines.replace("\t","");


            }
            review=review.trim();
            out.write(review+"\t"+lines);
            out.newLine();

        }
        buffer.close();
        out.close();
    }


    public void XMLCreate(String file) throws TransformerException {
        String textFile = file + "_ReviewText";
        int sentenceCount = 0;
        //Text_file="Output.txt";
        Scanner sc = null;
        try {
            DocumentBuilderFactory docFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder docBuilder = docFactory.newDocumentBuilder();
            sc = new Scanner(new File(file));
            String buf = null;
            Document doc = docBuilder.newDocument();
            Element rootElement = doc.createElement("sentences");
            doc.appendChild(rootElement);
            while (sc.hasNextLine()) {

                buf = sc.nextLine();
                sentenceCount++;

                String[] result0 = buf.split("\t");

                Element sentence = doc.createElement("sentence");
                rootElement.appendChild(sentence);
                Attr senAttr1 = doc.createAttribute("id");
                senAttr1.setValue(Integer.toString(sentenceCount));
                sentence.setAttributeNode(senAttr1);

                //System.out.println(sc.next());


                //temp2 = sc.nextLine();
                Element text = doc.createElement("text");

                text.appendChild(doc.createTextNode(result0[0].trim()));


                sentence.appendChild(text);

                if (result0.length > 1) {
                    Element aspectTerms = doc.createElement("aspectTerms");

                    sentence.appendChild(aspectTerms);
                    for (int j = 1; j < result0.length; j++) {
                        System.out.println(sentenceCount);
                        Element aspectTerm = doc.createElement("aspectTerm");
                        aspectTerms.appendChild(aspectTerm);
                        Attr aTAttr1 = doc.createAttribute("term");
                        aTAttr1.setValue(result0[j].trim());
                        aspectTerm.setAttributeNode(aTAttr1);

                        int t = 0;
                        int f = 0;
                        String to = "";
                        String from = "";

                        f = result0[0].indexOf(result0[j].trim());
                        from = Integer.toString(f);
                        t = f + result0[j].length();
                        to = Integer.toString(t);
                        Attr aTAttr3 = doc.createAttribute("from");
                        aTAttr3.setValue(from);
                        aspectTerm.setAttributeNode(aTAttr3);

                        Attr aTAttr4 = doc.createAttribute("to");
                        aTAttr4.setValue(to);
                        aspectTerm.setAttributeNode(aTAttr4);


                    }

                }

            }


            // write the content into xml file
            TransformerFactory transformerFactory = TransformerFactory.newInstance();
            Transformer transformer = transformerFactory.newTransformer();
            DOMSource source = new DOMSource(doc);
            StreamResult result = new StreamResult(new File(file + ".xml"));


            // StreamResult result2 = new StreamResult(System.out);

            transformer.transform(source, result);

            System.out.println("XML File Created :");

        } catch (ParserConfigurationException pce) {
            pce.printStackTrace();
        } catch (TransformerException tfe) {
            tfe.printStackTrace();
        } catch (ArrayIndexOutOfBoundsException e) {
            e.printStackTrace();
        } catch (IOException ex) {
            System.out.println(
                    "Error writing to file '"
                            + textFile + "'");
            // Or we could just do this:
            // ex.printStackTrace();
        }
    }
}
