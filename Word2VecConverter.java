package com.IITPatna.Project;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.UiServer;

import java.util.Collection;

/**
 * Created by GAURAV KUMAR on 7/18/2016.
 */
public class Word2VecConverter {



    public static void main(String[] args) throws Exception {

        Word2VecConverter word=new Word2VecConverter();
        //    args[0] specfiy the file name with space separated raw sentences to convert in vector form.
        word.Word2Vectors(args[0]);
    }

    public void Word2Vectors(String filePath) throws Exception {
        // String filePath = "raw_sentences.txt";


        System.out.println("Load & Vectorize Sentences....");
        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(filePath);
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());


        System.out.println("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(2)
                .iterations(10)
                .layerSize(100)
                .seed(42)
                .windowSize(10)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();


        System.out.println("Fitting Word2Vec model....");
        vec.fit();


        System.out.println("Writing word vectors to text file....");

        // Write word vectors
        WordVectorSerializer.writeWordVectors(vec, filePath+"pathToWriteto.txt");


        System.out.println("Closest Words:");

        Collection<String> lst = vec.wordsNearest("मुताबिक", 10);
        System.out.println(lst);
        UiServer server = UiServer.getInstance();
        System.out.println("Started on port " + server.getPort());
    }

}