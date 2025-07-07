package core;

import utils.*;
import discretization.*;
import feature_selection.*;

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
        System.out.println("\nTesting with synthetic Iris-like data...");
        
        // Create synthetic dataset similar to Iris
        DataEnhanced.setNumOfNumericalFeatures(4);
        DataEnhanced.setLabelInterval(3);
        
        DataSetEnhanced dataset = new DataSetEnhanced();
        
        // Add some synthetic data points
        dataset.addData(new DataEnhanced(new double[]{5.1, 3.5, 1.4, 0.2}, 1));
        dataset.addData(new DataEnhanced(new double[]{4.9, 3.0, 1.4, 0.2}, 1));
        dataset.addData(new DataEnhanced(new double[]{7.0, 3.2, 4.7, 1.4}, 2));
        dataset.addData(new DataEnhanced(new double[]{6.4, 3.2, 4.5, 1.5}, 2));
        dataset.addData(new DataEnhanced(new double[]{6.3, 3.3, 6.0, 2.5}, 3));
        dataset.addData(new DataEnhanced(new double[]{5.8, 2.7, 5.1, 1.9}, 3));
        
        System.out.println("Dataset created with " + dataset.getDataSetSize() + " samples");
        
        // Calculate feature statistics
        dataset.calculateFeatureStats();
        System.out.println("Feature statistics calculated");
        
        for (int i = 0; i < 4; i++) {
            System.out.printf("Feature %d: min=%.2f, max=%.2f, mean=%.2f, std=%.2f%n", 
                            i, dataset.getFeatureMin(i), dataset.getFeatureMax(i), 
                            dataset.getFeatureMean(i), dataset.getFeatureStd(i));
        }
        
        // Test discretization
        EnhancedDiscretizer discretizer = new EnhancedDiscretizer(dataset, config);
        double[][] cutPoints = discretizer.discretizeAllFeatures(
            EnhancedDiscretizer.DiscretizationMethod.ENTROPY_BASED);
        
        System.out.println("\nDiscretization completed:");
        for (int i = 0; i < cutPoints.length; i++) {
            System.out.printf("Feature %d: %d cut points%n", i, cutPoints[i].length);
        }
        
        // Test feature selection
        FeatureSelector featureSelector = new FeatureSelector(dataset, config);
        boolean[] selectedFeatures = featureSelector.selectFeatures(
            FeatureSelector.SelectionMethod.INFORMATION_GAIN, 3);
        
        System.out.println("\nFeature selection completed:");
        for (int i = 0; i < selectedFeatures.length; i++) {
            System.out.printf("Feature %d: %s%n", i, selectedFeatures[i] ? "SELECTED" : "not selected");
        }
        
        double[] importanceScores = featureSelector.getFeatureImportanceScores();
        System.out.println("\nFeature importance scores:");
        for (int i = 0; i < importanceScores.length; i++) {
            System.out.printf("Feature %d: %.4f%n", i, importanceScores[i]);
        }
        
        System.out.println("\nEnhanced MCRM basic functionality test completed successfully!");
    }
}