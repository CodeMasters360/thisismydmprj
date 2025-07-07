package core;

import utils.*;
import discretization.*;
import feature_selection.*;
import java.util.*;

/**
 * Main class for Enhanced MCRM Implementation
 */
public class EnhancedMain {
    
    public static void main(String[] args) {
        System.out.println("Enhanced MCRM Implementation");
        System.out.println("=============================");
        
        // Initialize configuration
        ConfigurationManager config = new ConfigurationManager();
        System.out.println("Configuration initialized with default parameters:");
        System.out.println("Colony Size: " + config.getColonySize());
        System.out.println("Max Epochs: " + config.getMaxEpochs());
        System.out.println("Feature Selection Probability: " + config.getFeatureSelectionProbability());
        
        // Test with synthetic data
        testWithSyntheticData(config);
    }
    
    private static void testWithSyntheticData(ConfigurationManager config) {
        System.out.println("\nTesting with enhanced synthetic Iris-like data...");
        
        // Create synthetic dataset similar to Iris
        DataEnhanced.setNumOfNumericalFeatures(4);
        DataEnhanced.setLabelInterval(3);
        
        DataSetEnhanced trainSet = createSyntheticTrainSet();
        DataSetEnhanced testSet = createSyntheticTestSet();
        
        System.out.printf("Created training set with %d samples%n", trainSet.getDataSetSize());
        System.out.printf("Created test set with %d samples%n", testSet.getDataSetSize());
        
        // Set reduced parameters for quick testing
        config.setParameter("maxEpochs", 10);
        config.setParameter("colonySize", 5);
        config.setParameter("archiveSize", 5);
        
        // Run the complete enhanced MCRM algorithm
        HybridMCRM enhancedMCRM = new HybridMCRM(config);
        enhancedMCRM.run(trainSet, testSet);
        
        // Display final results
        Map<String, Object> metrics = enhancedMCRM.getPerformanceMetrics();
        System.out.println("\n=== FINAL ENHANCED MCRM RESULTS ===");
        System.out.printf("Enhanced Accuracy: %.4f (%.2f%%)%n", 
                         (Double) metrics.get("accuracy"), (Double) metrics.get("accuracy") * 100);
        System.out.printf("Enhanced Coverage: %.4f%n", (Double) metrics.get("coverage"));
        System.out.printf("Enhanced Rules: %d%n", (Integer) metrics.get("numRules"));
        System.out.printf("Enhanced Avg Rule Length: %.2f%n", (Double) metrics.get("avgRuleLength"));
        
        System.out.println("\nEnhanced MCRM algorithm test completed successfully!");
    }
    
    private static DataSetEnhanced createSyntheticTrainSet() {
        DataSetEnhanced dataset = new DataSetEnhanced();
        
        // Class 1 (Setosa-like): Small values
        dataset.addData(new DataEnhanced(new double[]{5.1, 3.5, 1.4, 0.2}, 1));
        dataset.addData(new DataEnhanced(new double[]{4.9, 3.0, 1.4, 0.2}, 1));
        dataset.addData(new DataEnhanced(new double[]{4.7, 3.2, 1.3, 0.2}, 1));
        dataset.addData(new DataEnhanced(new double[]{5.0, 3.6, 1.4, 0.2}, 1));
        dataset.addData(new DataEnhanced(new double[]{5.4, 3.9, 1.7, 0.4}, 1));
        dataset.addData(new DataEnhanced(new double[]{4.6, 3.4, 1.4, 0.3}, 1));
        dataset.addData(new DataEnhanced(new double[]{5.0, 3.4, 1.5, 0.2}, 1));
        dataset.addData(new DataEnhanced(new double[]{4.4, 2.9, 1.4, 0.2}, 1));
        
        // Class 2 (Versicolor-like): Medium values  
        dataset.addData(new DataEnhanced(new double[]{7.0, 3.2, 4.7, 1.4}, 2));
        dataset.addData(new DataEnhanced(new double[]{6.4, 3.2, 4.5, 1.5}, 2));
        dataset.addData(new DataEnhanced(new double[]{6.9, 3.1, 4.9, 1.5}, 2));
        dataset.addData(new DataEnhanced(new double[]{5.5, 2.3, 4.0, 1.3}, 2));
        dataset.addData(new DataEnhanced(new double[]{6.5, 2.8, 4.6, 1.5}, 2));
        dataset.addData(new DataEnhanced(new double[]{5.7, 2.8, 4.5, 1.3}, 2));
        dataset.addData(new DataEnhanced(new double[]{6.3, 3.3, 4.7, 1.6}, 2));
        dataset.addData(new DataEnhanced(new double[]{4.9, 2.4, 3.3, 1.0}, 2));
        
        // Class 3 (Virginica-like): Large values
        dataset.addData(new DataEnhanced(new double[]{6.3, 3.3, 6.0, 2.5}, 3));
        dataset.addData(new DataEnhanced(new double[]{5.8, 2.7, 5.1, 1.9}, 3));
        dataset.addData(new DataEnhanced(new double[]{7.1, 3.0, 5.9, 2.1}, 3));
        dataset.addData(new DataEnhanced(new double[]{6.3, 2.9, 5.6, 1.8}, 3));
        dataset.addData(new DataEnhanced(new double[]{6.5, 3.0, 5.8, 2.2}, 3));
        dataset.addData(new DataEnhanced(new double[]{7.6, 3.0, 6.6, 2.1}, 3));
        dataset.addData(new DataEnhanced(new double[]{4.9, 2.5, 4.5, 1.7}, 3));
        dataset.addData(new DataEnhanced(new double[]{7.3, 2.9, 6.3, 1.8}, 3));
        
        return dataset;
    }
    
    private static DataSetEnhanced createSyntheticTestSet() {
        DataSetEnhanced dataset = new DataSetEnhanced();
        
        // Test samples for each class
        dataset.addData(new DataEnhanced(new double[]{5.2, 3.4, 1.5, 0.2}, 1));
        dataset.addData(new DataEnhanced(new double[]{4.8, 3.1, 1.3, 0.1}, 1));
        
        dataset.addData(new DataEnhanced(new double[]{6.1, 2.9, 4.3, 1.3}, 2));
        dataset.addData(new DataEnhanced(new double[]{5.9, 3.0, 4.2, 1.5}, 2));
        
        dataset.addData(new DataEnhanced(new double[]{6.2, 2.8, 5.4, 2.0}, 3));
        dataset.addData(new DataEnhanced(new double[]{7.0, 3.1, 5.8, 1.9}, 3));
        
        return dataset;
    }
}