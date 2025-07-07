package utils;

import java.util.ArrayList;
import java.util.Collections;

/**
 * Enhanced DataSet class with advanced data handling capabilities
 */
public class DataSetEnhanced {
    private ArrayList<DataEnhanced> data;
    private int dataSetSize;
    private double[][] featureStats; // [feature][min, max, mean, std]
    private double[][] correlationMatrix;
    
    public DataSetEnhanced() {
        this.data = new ArrayList<>();
        this.dataSetSize = 0;
    }
    
    public void addData(DataEnhanced d) {
        data.add(d);
        dataSetSize++;
    }
    
    public DataEnhanced getData(int index) {
        return data.get(index);
    }
    
    public int getDataSetSize() {
        return dataSetSize;
    }
    
    public ArrayList<DataEnhanced> getData() {
        return data;
    }
    
    /**
     * Calculate comprehensive feature statistics
     */
    public void calculateFeatureStats() {
        if (dataSetSize == 0) return;
        
        int numFeatures = DataEnhanced.getNumOfNumericalFeatures();
        featureStats = new double[numFeatures][4]; // min, max, mean, std
        
        // Initialize min/max
        for (int f = 0; f < numFeatures; f++) {
            featureStats[f][0] = Double.MAX_VALUE; // min
            featureStats[f][1] = -Double.MAX_VALUE; // max
            featureStats[f][2] = 0.0; // mean
            featureStats[f][3] = 0.0; // std
        }
        
        // Calculate min, max, and sum for mean
        for (int i = 0; i < dataSetSize; i++) {
            for (int f = 0; f < numFeatures; f++) {
                double value = data.get(i).numericalFeatures[f];
                if (value < featureStats[f][0]) featureStats[f][0] = value;
                if (value > featureStats[f][1]) featureStats[f][1] = value;
                featureStats[f][2] += value;
            }
        }
        
        // Calculate mean
        for (int f = 0; f < numFeatures; f++) {
            featureStats[f][2] /= dataSetSize;
        }
        
        // Calculate standard deviation
        for (int i = 0; i < dataSetSize; i++) {
            for (int f = 0; f < numFeatures; f++) {
                double diff = data.get(i).numericalFeatures[f] - featureStats[f][2];
                featureStats[f][3] += diff * diff;
            }
        }
        
        for (int f = 0; f < numFeatures; f++) {
            featureStats[f][3] = Math.sqrt(featureStats[f][3] / dataSetSize);
        }
    }
    
    /**
     * Calculate correlation matrix between features
     */
    public void calculateCorrelationMatrix() {
        if (dataSetSize == 0) return;
        
        int numFeatures = DataEnhanced.getNumOfNumericalFeatures();
        correlationMatrix = new double[numFeatures][numFeatures];
        
        if (featureStats == null) {
            calculateFeatureStats();
        }
        
        for (int i = 0; i < numFeatures; i++) {
            for (int j = 0; j < numFeatures; j++) {
                if (i == j) {
                    correlationMatrix[i][j] = 1.0;
                } else {
                    double covariance = 0.0;
                    for (int k = 0; k < dataSetSize; k++) {
                        double xi = data.get(k).numericalFeatures[i] - featureStats[i][2];
                        double xj = data.get(k).numericalFeatures[j] - featureStats[j][2];
                        covariance += xi * xj;
                    }
                    covariance /= dataSetSize;
                    
                    double correlation = covariance / (featureStats[i][3] * featureStats[j][3]);
                    correlationMatrix[i][j] = correlation;
                }
            }
        }
    }
    
    // Getters
    public double[][] getFeatureStats() {
        return featureStats;
    }
    
    public double[][] getCorrelationMatrix() {
        return correlationMatrix;
    }
    
    public double getFeatureMin(int feature) {
        return featureStats != null ? featureStats[feature][0] : 0.0;
    }
    
    public double getFeatureMax(int feature) {
        return featureStats != null ? featureStats[feature][1] : 0.0;
    }
    
    public double getFeatureMean(int feature) {
        return featureStats != null ? featureStats[feature][2] : 0.0;
    }
    
    public double getFeatureStd(int feature) {
        return featureStats != null ? featureStats[feature][3] : 0.0;
    }
}