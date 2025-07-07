package feature_selection;

import utils.DataSetEnhanced;
import utils.DataEnhanced;
import utils.ConfigurationManager;
import java.util.*;

/**
 * Multi-method intelligent feature selection with correlation analysis
 */
public class FeatureSelector {
    private DataSetEnhanced dataset;
    private ConfigurationManager config;
    private double[] featureImportanceScores;
    private boolean[] selectedFeatures;
    
    // Feature selection methods
    public enum SelectionMethod {
        INFORMATION_GAIN,
        CHI_SQUARE,
        VARIANCE_BASED,
        CORRELATION_BASED,
        MUTUAL_INFORMATION,
        RELIEF_F,
        HYBRID
    }
    
    public FeatureSelector(DataSetEnhanced dataset, ConfigurationManager config) {
        this.dataset = dataset;
        this.config = config;
        int numFeatures = DataEnhanced.getNumOfNumericalFeatures();
        this.featureImportanceScores = new double[numFeatures];
        this.selectedFeatures = new boolean[numFeatures];
        
        // Initialize all features as selected
        Arrays.fill(selectedFeatures, true);
    }
    
    /**
     * Select features using the specified method
     */
    public boolean[] selectFeatures(SelectionMethod method, int maxFeatures) {
        calculateFeatureImportance(method);
        return selectTopFeatures(maxFeatures);
    }
    
    /**
     * Calculate feature importance scores using different methods
     */
    public void calculateFeatureImportance(SelectionMethod method) {
        switch (method) {
            case INFORMATION_GAIN:
                calculateInformationGain();
                break;
            case CHI_SQUARE:
                calculateChiSquareScores();
                break;
            case VARIANCE_BASED:
                calculateVarianceScores();
                break;
            case CORRELATION_BASED:
                calculateCorrelationScores();
                break;
            case MUTUAL_INFORMATION:
                calculateMutualInformation();
                break;
            case RELIEF_F:
                calculateReliefFScores();
                break;
            case HYBRID:
                calculateHybridScores();
                break;
        }
    }
    
    /**
     * Information Gain based feature importance
     */
    private void calculateInformationGain() {
        double classEntropy = calculateClassEntropy(dataset.getData());
        
        for (int feature = 0; feature < DataEnhanced.getNumOfNumericalFeatures(); feature++) {
            double conditionalEntropy = calculateConditionalEntropy(feature);
            featureImportanceScores[feature] = classEntropy - conditionalEntropy;
        }
        
        normalizeScores();
    }
    
    /**
     * Chi-square based feature importance
     */
    private void calculateChiSquareScores() {
        for (int feature = 0; feature < DataEnhanced.getNumOfNumericalFeatures(); feature++) {
            featureImportanceScores[feature] = calculateChiSquareForFeature(feature);
        }
        
        normalizeScores();
    }
    
    /**
     * Variance-based feature importance
     */
    private void calculateVarianceScores() {
        dataset.calculateFeatureStats();
        
        for (int feature = 0; feature < DataEnhanced.getNumOfNumericalFeatures(); feature++) {
            double variance = Math.pow(dataset.getFeatureStd(feature), 2);
            featureImportanceScores[feature] = variance;
        }
        
        normalizeScores();
    }
    
    /**
     * Correlation-based feature importance
     */
    private void calculateCorrelationScores() {
        // Calculate correlation with class labels and feature redundancy
        for (int feature = 0; feature < DataEnhanced.getNumOfNumericalFeatures(); feature++) {
            double classCorrelation = calculateClassCorrelation(feature);
            double redundancy = calculateFeatureRedundancy(feature);
            
            // Merit = class_correlation / (1 + redundancy)
            featureImportanceScores[feature] = Math.abs(classCorrelation) / (1.0 + redundancy);
        }
        
        normalizeScores();
    }
    
    /**
     * Mutual Information based feature importance
     */
    private void calculateMutualInformation() {
        for (int feature = 0; feature < DataEnhanced.getNumOfNumericalFeatures(); feature++) {
            featureImportanceScores[feature] = calculateMutualInfoForFeature(feature);
        }
        
        normalizeScores();
    }
    
    /**
     * ReliefF algorithm for feature importance
     */
    private void calculateReliefFScores() {
        int numSamples = Math.min(dataset.getDataSetSize(), 100); // Sample size
        Arrays.fill(featureImportanceScores, 0.0);
        
        Random random = new Random(42);
        
        for (int i = 0; i < numSamples; i++) {
            // Select random instance
            int randomIndex = random.nextInt(dataset.getDataSetSize());
            DataEnhanced randomInstance = dataset.getData(randomIndex);
            
            // Find nearest hit and nearest miss
            DataEnhanced nearestHit = findNearestHit(randomInstance);
            DataEnhanced nearestMiss = findNearestMiss(randomInstance);
            
            if (nearestHit != null && nearestMiss != null) {
                // Update feature weights
                for (int feature = 0; feature < DataEnhanced.getNumOfNumericalFeatures(); feature++) {
                    double diffHit = Math.abs(randomInstance.numericalFeatures[feature] - 
                                            nearestHit.numericalFeatures[feature]);
                    double diffMiss = Math.abs(randomInstance.numericalFeatures[feature] - 
                                             nearestMiss.numericalFeatures[feature]);
                    
                    featureImportanceScores[feature] += (diffMiss - diffHit) / numSamples;
                }
            }
        }
        
        normalizeScores();
    }
    
    /**
     * Hybrid feature selection combining multiple methods
     */
    private void calculateHybridScores() {
        // Calculate scores using multiple methods
        double[] igScores = new double[DataEnhanced.getNumOfNumericalFeatures()];
        double[] chiScores = new double[DataEnhanced.getNumOfNumericalFeatures()];
        double[] corrScores = new double[DataEnhanced.getNumOfNumericalFeatures()];
        
        // Store original scores
        System.arraycopy(featureImportanceScores, 0, igScores, 0, featureImportanceScores.length);
        
        calculateInformationGain();
        System.arraycopy(featureImportanceScores, 0, igScores, 0, featureImportanceScores.length);
        
        calculateChiSquareScores();
        System.arraycopy(featureImportanceScores, 0, chiScores, 0, featureImportanceScores.length);
        
        calculateCorrelationScores();
        System.arraycopy(featureImportanceScores, 0, corrScores, 0, featureImportanceScores.length);
        
        // Combine scores with weights
        for (int i = 0; i < featureImportanceScores.length; i++) {
            featureImportanceScores[i] = 0.4 * igScores[i] + 0.3 * chiScores[i] + 0.3 * corrScores[i];
        }
    }
    
    /**
     * Select top features based on importance scores
     */
    private boolean[] selectTopFeatures(int maxFeatures) {
        // Create feature-score pairs
        List<FeatureScore> featureScores = new ArrayList<>();
        for (int i = 0; i < featureImportanceScores.length; i++) {
            featureScores.add(new FeatureScore(i, featureImportanceScores[i]));
        }
        
        // Sort by score descending
        featureScores.sort((a, b) -> Double.compare(b.score, a.score));
        
        // Select top features
        Arrays.fill(selectedFeatures, false);
        int numToSelect = Math.min(maxFeatures, featureScores.size());
        
        for (int i = 0; i < numToSelect; i++) {
            selectedFeatures[featureScores.get(i).featureIndex] = true;
        }
        
        return selectedFeatures.clone();
    }
    
    /**
     * Remove highly correlated features (feature redundancy removal)
     */
    public boolean[] removeRedundantFeatures(double correlationThreshold) {
        dataset.calculateCorrelationMatrix();
        double[][] correlationMatrix = dataset.getCorrelationMatrix();
        
        boolean[] toRemove = new boolean[DataEnhanced.getNumOfNumericalFeatures()];
        
        for (int i = 0; i < correlationMatrix.length; i++) {
            for (int j = i + 1; j < correlationMatrix[i].length; j++) {
                if (Math.abs(correlationMatrix[i][j]) > correlationThreshold) {
                    // Remove the feature with lower importance
                    if (featureImportanceScores[i] < featureImportanceScores[j]) {
                        toRemove[i] = true;
                    } else {
                        toRemove[j] = true;
                    }
                }
            }
        }
        
        // Update selected features
        for (int i = 0; i < selectedFeatures.length; i++) {
            if (toRemove[i]) {
                selectedFeatures[i] = false;
            }
        }
        
        return selectedFeatures.clone();
    }
    
    // Helper methods
    private double calculateClassEntropy(List<DataEnhanced> data) {
        if (data.isEmpty()) return 0.0;
        
        Map<Integer, Integer> classCount = new HashMap<>();
        for (DataEnhanced d : data) {
            classCount.put(d.label, classCount.getOrDefault(d.label, 0) + 1);
        }
        
        double entropy = 0.0;
        int total = data.size();
        
        for (int count : classCount.values()) {
            if (count > 0) {
                double probability = (double) count / total;
                entropy -= probability * Math.log(probability) / Math.log(2);
            }
        }
        
        return entropy;
    }
    
    private double calculateConditionalEntropy(int featureIndex) {
        // Discretize feature into bins for entropy calculation
        double min = dataset.getFeatureMin(featureIndex);
        double max = dataset.getFeatureMax(featureIndex);
        int numBins = 10; // Can be made configurable
        
        Map<Integer, List<DataEnhanced>> bins = new HashMap<>();
        
        for (DataEnhanced data : dataset.getData()) {
            double value = data.numericalFeatures[featureIndex];
            int binIndex = (int) ((value - min) / (max - min) * (numBins - 1));
            binIndex = Math.max(0, Math.min(numBins - 1, binIndex));
            
            bins.computeIfAbsent(binIndex, k -> new ArrayList<>()).add(data);
        }
        
        double conditionalEntropy = 0.0;
        int totalSize = dataset.getDataSetSize();
        
        for (List<DataEnhanced> bin : bins.values()) {
            if (!bin.isEmpty()) {
                double weight = (double) bin.size() / totalSize;
                double entropy = calculateClassEntropy(bin);
                conditionalEntropy += weight * entropy;
            }
        }
        
        return conditionalEntropy;
    }
    
    private double calculateChiSquareForFeature(int featureIndex) {
        // Simplified chi-square calculation
        // In practice, this would involve proper contingency table analysis
        double chiSquare = 0.0;
        
        // Create contingency table for feature bins vs class labels
        Map<Integer, Map<Integer, Integer>> contingencyTable = createContingencyTable(featureIndex);
        
        // Calculate chi-square statistic
        for (Map<Integer, Integer> binCounts : contingencyTable.values()) {
            for (int count : binCounts.values()) {
                // Chi-square calculation would go here
                // This is a simplified placeholder
                chiSquare += count * 0.1; // Placeholder
            }
        }
        
        return chiSquare;
    }
    
    private Map<Integer, Map<Integer, Integer>> createContingencyTable(int featureIndex) {
        Map<Integer, Map<Integer, Integer>> table = new HashMap<>();
        
        // Simplified implementation
        for (DataEnhanced data : dataset.getData()) {
            int binIndex = 0; // Simplified binning
            table.computeIfAbsent(binIndex, k -> new HashMap<>())
                 .put(data.label, table.get(binIndex).getOrDefault(data.label, 0) + 1);
        }
        
        return table;
    }
    
    private double calculateClassCorrelation(int featureIndex) {
        // Calculate Pearson correlation between feature and class labels
        double[] featureValues = new double[dataset.getDataSetSize()];
        double[] classLabels = new double[dataset.getDataSetSize()];
        
        for (int i = 0; i < dataset.getDataSetSize(); i++) {
            featureValues[i] = dataset.getData(i).numericalFeatures[featureIndex];
            classLabels[i] = dataset.getData(i).label;
        }
        
        return calculatePearsonCorrelation(featureValues, classLabels);
    }
    
    private double calculateFeatureRedundancy(int featureIndex) {
        double maxCorrelation = 0.0;
        dataset.calculateCorrelationMatrix();
        double[][] correlationMatrix = dataset.getCorrelationMatrix();
        
        for (int i = 0; i < correlationMatrix[featureIndex].length; i++) {
            if (i != featureIndex && selectedFeatures[i]) {
                maxCorrelation = Math.max(maxCorrelation, 
                                        Math.abs(correlationMatrix[featureIndex][i]));
            }
        }
        
        return maxCorrelation;
    }
    
    private double calculateMutualInfoForFeature(int featureIndex) {
        // Simplified mutual information calculation
        // In practice, this would involve proper discretization and probability estimation
        return calculateConditionalEntropy(featureIndex) * 0.5; // Placeholder
    }
    
    private DataEnhanced findNearestHit(DataEnhanced instance) {
        DataEnhanced nearest = null;
        double minDistance = Double.MAX_VALUE;
        
        for (DataEnhanced data : dataset.getData()) {
            if (data != instance && data.label == instance.label) {
                double distance = instance.distance(data);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearest = data;
                }
            }
        }
        
        return nearest;
    }
    
    private DataEnhanced findNearestMiss(DataEnhanced instance) {
        DataEnhanced nearest = null;
        double minDistance = Double.MAX_VALUE;
        
        for (DataEnhanced data : dataset.getData()) {
            if (data != instance && data.label != instance.label) {
                double distance = instance.distance(data);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearest = data;
                }
            }
        }
        
        return nearest;
    }
    
    private double calculatePearsonCorrelation(double[] x, double[] y) {
        if (x.length != y.length) return 0.0;
        
        double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0, sumY2 = 0.0;
        int n = x.length;
        
        for (int i = 0; i < n; i++) {
            sumX += x[i];
            sumY += y[i];
            sumXY += x[i] * y[i];
            sumX2 += x[i] * x[i];
            sumY2 += y[i] * y[i];
        }
        
        double numerator = n * sumXY - sumX * sumY;
        double denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        return denominator != 0 ? numerator / denominator : 0.0;
    }
    
    private void normalizeScores() {
        double max = Arrays.stream(featureImportanceScores).max().orElse(1.0);
        if (max > 0) {
            for (int i = 0; i < featureImportanceScores.length; i++) {
                featureImportanceScores[i] /= max;
            }
        }
    }
    
    // Getters
    public double[] getFeatureImportanceScores() {
        return featureImportanceScores.clone();
    }
    
    public boolean[] getSelectedFeatures() {
        return selectedFeatures.clone();
    }
    
    public int getNumSelectedFeatures() {
        int count = 0;
        for (boolean selected : selectedFeatures) {
            if (selected) count++;
        }
        return count;
    }
    
    // Inner class for feature-score pairs
    private static class FeatureScore {
        int featureIndex;
        double score;
        
        FeatureScore(int featureIndex, double score) {
            this.featureIndex = featureIndex;
            this.score = score;
        }
    }
}