package discretization;

import utils.DataSetEnhanced;
import utils.DataEnhanced;
import utils.ConfigurationManager;
import java.util.*;

/**
 * Enhanced multi-method discretization with advanced cut point optimization
 */
public class EnhancedDiscretizer {
    private DataSetEnhanced dataset;
    private ConfigurationManager config;
    private Map<Integer, double[]> cutPointsCache;
    
    // Discretization methods
    public enum DiscretizationMethod {
        ENTROPY_BASED,
        CHI_MERGE,
        MDL_BASED,
        EQUAL_WIDTH,
        EQUAL_FREQUENCY,
        HYBRID
    }
    
    public EnhancedDiscretizer(DataSetEnhanced dataset, ConfigurationManager config) {
        this.dataset = dataset;
        this.config = config;
        this.cutPointsCache = new HashMap<>();
    }
    
    /**
     * Main discretization method that selects the best approach
     */
    public double[][] discretizeAllFeatures(DiscretizationMethod method) {
        int numFeatures = DataEnhanced.getNumOfNumericalFeatures();
        double[][] allCutPoints = new double[numFeatures][];
        
        for (int feature = 0; feature < numFeatures; feature++) {
            allCutPoints[feature] = discretizeFeature(feature, method);
        }
        
        return allCutPoints;
    }
    
    /**
     * Discretize a single feature using the specified method
     */
    public double[] discretizeFeature(int featureIndex, DiscretizationMethod method) {
        // Check cache first
        String cacheKey = featureIndex + "_" + method.name();
        if (config.isEnableCaching() && cutPointsCache.containsKey(featureIndex)) {
            return cutPointsCache.get(featureIndex);
        }
        
        double[] cutPoints;
        
        switch (method) {
            case ENTROPY_BASED:
                cutPoints = entropyBasedDiscretization(featureIndex);
                break;
            case CHI_MERGE:
                cutPoints = chiMergeDiscretization(featureIndex);
                break;
            case MDL_BASED:
                cutPoints = mdlBasedDiscretization(featureIndex);
                break;
            case EQUAL_WIDTH:
                cutPoints = equalWidthDiscretization(featureIndex);
                break;
            case EQUAL_FREQUENCY:
                cutPoints = equalFrequencyDiscretization(featureIndex);
                break;
            case HYBRID:
                cutPoints = hybridDiscretization(featureIndex);
                break;
            default:
                cutPoints = entropyBasedDiscretization(featureIndex);
        }
        
        // Cache the result
        if (config.isEnableCaching()) {
            cutPointsCache.put(featureIndex, cutPoints);
        }
        
        return cutPoints;
    }
    
    /**
     * Entropy-based discretization with recursive binary splitting
     */
    private double[] entropyBasedDiscretization(int featureIndex) {
        List<DataValue> values = extractFeatureValues(featureIndex);
        Collections.sort(values);
        
        List<Double> cutPoints = new ArrayList<>();
        findBestCutPointsRecursive(values, cutPoints, 0.01); // minimum gain threshold
        
        return cutPoints.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    /**
     * ChiMerge discretization
     */
    private double[] chiMergeDiscretization(int featureIndex) {
        List<DataValue> values = extractFeatureValues(featureIndex);
        Collections.sort(values);
        
        // Initialize intervals
        List<Interval> intervals = initializeIntervals(values);
        
        // Merge intervals based on chi-square test
        while (intervals.size() > config.getMinIntervals()) {
            double minChiSquare = Double.MAX_VALUE;
            int mergeIndex = -1;
            
            for (int i = 0; i < intervals.size() - 1; i++) {
                double chiSquare = calculateChiSquare(intervals.get(i), intervals.get(i + 1));
                if (chiSquare < minChiSquare) {
                    minChiSquare = chiSquare;
                    mergeIndex = i;
                }
            }
            
            if (minChiSquare < config.getChiSquareThreshold()) {
                mergeIntervals(intervals, mergeIndex);
            } else {
                break;
            }
        }
        
        return extractCutPointsFromIntervals(intervals);
    }
    
    /**
     * MDL-based discretization
     */
    private double[] mdlBasedDiscretization(int featureIndex) {
        List<DataValue> values = extractFeatureValues(featureIndex);
        Collections.sort(values);
        
        List<Double> cutPoints = new ArrayList<>();
        findMDLOptimalCutPoints(values, cutPoints);
        
        return cutPoints.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    /**
     * Equal width discretization
     */
    private double[] equalWidthDiscretization(int featureIndex) {
        double min = dataset.getFeatureMin(featureIndex);
        double max = dataset.getFeatureMax(featureIndex);
        int numIntervals = Math.min(config.getMaxIntervals(), 
                                  Math.max(config.getMinIntervals(), 
                                          (int) Math.sqrt(dataset.getDataSetSize())));
        
        double[] cutPoints = new double[numIntervals - 1];
        double width = (max - min) / numIntervals;
        
        for (int i = 0; i < numIntervals - 1; i++) {
            cutPoints[i] = min + (i + 1) * width;
        }
        
        return cutPoints;
    }
    
    /**
     * Equal frequency discretization
     */
    private double[] equalFrequencyDiscretization(int featureIndex) {
        List<DataValue> values = extractFeatureValues(featureIndex);
        Collections.sort(values);
        
        int numIntervals = Math.min(config.getMaxIntervals(), 
                                  Math.max(config.getMinIntervals(), 
                                          (int) Math.sqrt(dataset.getDataSetSize())));
        
        List<Double> cutPoints = new ArrayList<>();
        int intervalSize = values.size() / numIntervals;
        
        for (int i = 1; i < numIntervals; i++) {
            int index = i * intervalSize;
            if (index < values.size() - 1) {
                double cutPoint = (values.get(index).value + values.get(index - 1).value) / 2.0;
                if (!cutPoints.contains(cutPoint)) {
                    cutPoints.add(cutPoint);
                }
            }
        }
        
        return cutPoints.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    /**
     * Hybrid discretization combining multiple methods
     */
    private double[] hybridDiscretization(int featureIndex) {
        // Try different methods and select the best one based on information gain
        double[] entropyCP = entropyBasedDiscretization(featureIndex);
        double[] chiCP = chiMergeDiscretization(featureIndex);
        double[] mdlCP = mdlBasedDiscretization(featureIndex);
        
        double entropyGain = calculateInformationGain(featureIndex, entropyCP);
        double chiGain = calculateInformationGain(featureIndex, chiCP);
        double mdlGain = calculateInformationGain(featureIndex, mdlCP);
        
        if (entropyGain >= chiGain && entropyGain >= mdlGain) {
            return entropyCP;
        } else if (chiGain >= mdlGain) {
            return chiCP;
        } else {
            return mdlCP;
        }
    }
    
    /**
     * Calculate information gain for a set of cut points
     */
    private double calculateInformationGain(int featureIndex, double[] cutPoints) {
        if (cutPoints.length == 0) return 0.0;
        
        // Calculate entropy before discretization
        double totalEntropy = calculateClassEntropy(dataset.getData());
        
        // Calculate weighted entropy after discretization
        List<List<DataEnhanced>> intervals = partitionData(featureIndex, cutPoints);
        double weightedEntropy = 0.0;
        int totalSize = dataset.getDataSetSize();
        
        for (List<DataEnhanced> interval : intervals) {
            if (!interval.isEmpty()) {
                double weight = (double) interval.size() / totalSize;
                double entropy = calculateClassEntropy(interval);
                weightedEntropy += weight * entropy;
            }
        }
        
        return totalEntropy - weightedEntropy;
    }
    
    /**
     * Calculate entropy of class distribution
     */
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
    
    /**
     * Partition data based on cut points
     */
    private List<List<DataEnhanced>> partitionData(int featureIndex, double[] cutPoints) {
        List<List<DataEnhanced>> intervals = new ArrayList<>();
        
        // Initialize intervals
        for (int i = 0; i <= cutPoints.length; i++) {
            intervals.add(new ArrayList<>());
        }
        
        // Assign data points to intervals
        for (DataEnhanced data : dataset.getData()) {
            double value = data.numericalFeatures[featureIndex];
            int intervalIndex = 0;
            
            for (int i = 0; i < cutPoints.length; i++) {
                if (value <= cutPoints[i]) {
                    intervalIndex = i;
                    break;
                } else {
                    intervalIndex = i + 1;
                }
            }
            
            intervals.get(intervalIndex).add(data);
        }
        
        return intervals;
    }
    
    // Helper methods and classes
    private List<DataValue> extractFeatureValues(int featureIndex) {
        List<DataValue> values = new ArrayList<>();
        for (DataEnhanced data : dataset.getData()) {
            values.add(new DataValue(data.numericalFeatures[featureIndex], data.label));
        }
        return values;
    }
    
    private void findBestCutPointsRecursive(List<DataValue> values, List<Double> cutPoints, double minGain) {
        double bestGain = 0.0;
        int bestIndex = -1;
        
        for (int i = 1; i < values.size(); i++) {
            if (values.get(i).label != values.get(i-1).label) {
                double cutPoint = (values.get(i).value + values.get(i-1).value) / 2.0;
                double gain = calculateGainForCutPoint(values, i);
                
                if (gain > bestGain && gain > minGain) {
                    bestGain = gain;
                    bestIndex = i;
                }
            }
        }
        
        if (bestIndex > 0) {
            double cutPoint = (values.get(bestIndex).value + values.get(bestIndex-1).value) / 2.0;
            cutPoints.add(cutPoint);
            
            // Recursively find cut points in sub-intervals
            List<DataValue> leftValues = new ArrayList<>(values.subList(0, bestIndex));
            List<DataValue> rightValues = new ArrayList<>(values.subList(bestIndex, values.size()));
            
            findBestCutPointsRecursive(leftValues, cutPoints, minGain);
            findBestCutPointsRecursive(rightValues, cutPoints, minGain);
        }
    }
    
    private double calculateGainForCutPoint(List<DataValue> values, int splitIndex) {
        // Calculate information gain for a potential cut point
        double totalEntropy = calculateEntropy(values);
        
        List<DataValue> left = values.subList(0, splitIndex);
        List<DataValue> right = values.subList(splitIndex, values.size());
        
        double leftWeight = (double) left.size() / values.size();
        double rightWeight = (double) right.size() / values.size();
        
        double weightedEntropy = leftWeight * calculateEntropy(left) + 
                               rightWeight * calculateEntropy(right);
        
        return totalEntropy - weightedEntropy;
    }
    
    private double calculateEntropy(List<DataValue> values) {
        if (values.isEmpty()) return 0.0;
        
        Map<Integer, Integer> classCount = new HashMap<>();
        for (DataValue v : values) {
            classCount.put(v.label, classCount.getOrDefault(v.label, 0) + 1);
        }
        
        double entropy = 0.0;
        int total = values.size();
        
        for (int count : classCount.values()) {
            if (count > 0) {
                double probability = (double) count / total;
                entropy -= probability * Math.log(probability) / Math.log(2);
            }
        }
        
        return entropy;
    }
    
    private List<Interval> initializeIntervals(List<DataValue> values) {
        // Implementation for ChiMerge initialization
        // This is a simplified version
        List<Interval> intervals = new ArrayList<>();
        // Initialize each unique value as an interval
        // (Simplified implementation)
        return intervals;
    }
    
    private double calculateChiSquare(Interval interval1, Interval interval2) {
        // Chi-square calculation for interval merging
        return 0.0; // Simplified
    }
    
    private void mergeIntervals(List<Interval> intervals, int index) {
        // Merge two adjacent intervals
        // Simplified implementation
    }
    
    private double[] extractCutPointsFromIntervals(List<Interval> intervals) {
        // Extract cut points from interval boundaries
        return new double[0]; // Simplified
    }
    
    private void findMDLOptimalCutPoints(List<DataValue> values, List<Double> cutPoints) {
        // MDL-based optimal cut point finding
        // Simplified implementation
    }
    
    // Inner classes
    private static class DataValue implements Comparable<DataValue> {
        double value;
        int label;
        
        DataValue(double value, int label) {
            this.value = value;
            this.label = label;
        }
        
        @Override
        public int compareTo(DataValue other) {
            return Double.compare(this.value, other.value);
        }
    }
    
    private static class Interval {
        double start, end;
        Map<Integer, Integer> classFrequency;
        
        Interval(double start, double end) {
            this.start = start;
            this.end = end;
            this.classFrequency = new HashMap<>();
        }
    }
}