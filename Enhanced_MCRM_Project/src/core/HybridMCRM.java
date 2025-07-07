package core;

import utils.*;
import discretization.*;
import feature_selection.*;
import java.util.*;

/**
 * Enhanced MCRM Algorithm - Main hybrid implementation
 */
public class HybridMCRM {
    private ConfigurationManager config;
    private DataSetEnhanced trainSet;
    private DataSetEnhanced testSet;
    private EnhancedDiscretizer discretizer;
    private FeatureSelector featureSelector;
    private EnhancedABC abc;
    
    // Algorithm state
    private double[][] cutPoints;
    private boolean[] selectedFeatures;
    private List<EnhancedArea> finalRules;
    private Map<String, Object> performanceMetrics;
    
    public HybridMCRM(ConfigurationManager config) {
        this.config = config;
        this.performanceMetrics = new HashMap<>();
    }
    
    /**
     * Main algorithm execution
     */
    public void run(DataSetEnhanced trainSet, DataSetEnhanced testSet) {
        this.trainSet = trainSet;
        this.testSet = testSet;
        
        System.out.println("Starting Enhanced MCRM Algorithm...");
        
        // Phase 1: Data preprocessing and feature analysis
        preprocessData();
        
        // Phase 2: Advanced discretization
        performDiscretization();
        
        // Phase 3: Intelligent feature selection
        performFeatureSelection();
        
        // Phase 4: Enhanced ABC optimization
        performOptimization();
        
        // Phase 5: Rule extraction and refinement
        extractAndRefineRules();
        
        // Phase 6: Comprehensive evaluation
        evaluatePerformance();
        
        System.out.println("Enhanced MCRM Algorithm completed!");
    }
    
    private void preprocessData() {
        System.out.println("Phase 1: Data preprocessing...");
        
        // Calculate comprehensive statistics
        trainSet.calculateFeatureStats();
        trainSet.calculateCorrelationMatrix();
        
        // Display data characteristics
        System.out.printf("Training set: %d instances, %d features%n", 
                         trainSet.getDataSetSize(), DataEnhanced.getNumOfNumericalFeatures());
        
        for (int i = 0; i < DataEnhanced.getNumOfNumericalFeatures(); i++) {
            System.out.printf("Feature %d: min=%.3f, max=%.3f, mean=%.3f, std=%.3f%n", 
                            i, trainSet.getFeatureMin(i), trainSet.getFeatureMax(i), 
                            trainSet.getFeatureMean(i), trainSet.getFeatureStd(i));
        }
    }
    
    private void performDiscretization() {
        System.out.println("Phase 2: Advanced discretization...");
        
        discretizer = new EnhancedDiscretizer(trainSet, config);
        
        // Try different discretization methods and select the best
        EnhancedDiscretizer.DiscretizationMethod[] methods = {
            EnhancedDiscretizer.DiscretizationMethod.ENTROPY_BASED,
            EnhancedDiscretizer.DiscretizationMethod.CHI_MERGE,
            EnhancedDiscretizer.DiscretizationMethod.HYBRID
        };
        
        double bestScore = -1.0;
        double[][] bestCutPoints = null;
        
        for (EnhancedDiscretizer.DiscretizationMethod method : methods) {
            double[][] candidateCutPoints = discretizer.discretizeAllFeatures(method);
            double score = evaluateDiscretization(candidateCutPoints);
            
            System.out.printf("Discretization method %s: score=%.4f%n", method, score);
            
            if (score > bestScore) {
                bestScore = score;
                bestCutPoints = candidateCutPoints;
            }
        }
        
        this.cutPoints = bestCutPoints;
        System.out.printf("Selected discretization with score: %.4f%n", bestScore);
        
        // Display discretization results
        for (int i = 0; i < cutPoints.length; i++) {
            System.out.printf("Feature %d: %d intervals%n", i, cutPoints[i].length + 1);
        }
    }
    
    private double evaluateDiscretization(double[][] cutPoints) {
        // Simple evaluation based on information gain
        double totalGain = 0.0;
        
        for (int feature = 0; feature < cutPoints.length; feature++) {
            if (cutPoints[feature].length > 0) {
                totalGain += calculateInformationGainForFeature(feature, cutPoints[feature]);
            }
        }
        
        return totalGain / cutPoints.length;
    }
    
    private double calculateInformationGainForFeature(int featureIndex, double[] featureCutPoints) {
        // Simplified information gain calculation
        if (featureCutPoints.length == 0) return 0.0;
        
        // Calculate entropy before discretization
        Map<Integer, Integer> classCount = new HashMap<>();
        for (int i = 0; i < trainSet.getDataSetSize(); i++) {
            int label = trainSet.getData(i).label;
            classCount.put(label, classCount.getOrDefault(label, 0) + 1);
        }
        
        double totalEntropy = calculateEntropy(classCount, trainSet.getDataSetSize());
        
        // Calculate weighted entropy after discretization
        double weightedEntropy = 0.0;
        
        for (int interval = 0; interval <= featureCutPoints.length; interval++) {
            Map<Integer, Integer> intervalClassCount = new HashMap<>();
            int intervalSize = 0;
            
            for (int i = 0; i < trainSet.getDataSetSize(); i++) {
                double value = trainSet.getData(i).numericalFeatures[featureIndex];
                boolean inInterval = false;
                
                if (interval == 0) {
                    inInterval = (featureCutPoints.length == 0) || (value <= featureCutPoints[0]);
                } else if (interval == featureCutPoints.length) {
                    inInterval = value > featureCutPoints[interval - 1];
                } else {
                    inInterval = value > featureCutPoints[interval - 1] && value <= featureCutPoints[interval];
                }
                
                if (inInterval) {
                    int label = trainSet.getData(i).label;
                    intervalClassCount.put(label, intervalClassCount.getOrDefault(label, 0) + 1);
                    intervalSize++;
                }
            }
            
            if (intervalSize > 0) {
                double intervalEntropy = calculateEntropy(intervalClassCount, intervalSize);
                weightedEntropy += ((double) intervalSize / trainSet.getDataSetSize()) * intervalEntropy;
            }
        }
        
        return totalEntropy - weightedEntropy;
    }
    
    private double calculateEntropy(Map<Integer, Integer> classCounts, int total) {
        if (total == 0) return 0.0;
        
        double entropy = 0.0;
        for (int count : classCounts.values()) {
            if (count > 0) {
                double probability = (double) count / total;
                entropy -= probability * Math.log(probability) / Math.log(2);
            }
        }
        
        return entropy;
    }
    
    private void performFeatureSelection() {
        System.out.println("Phase 3: Intelligent feature selection...");
        
        featureSelector = new FeatureSelector(trainSet, config);
        
        // Calculate feature importance using multiple methods
        FeatureSelector.SelectionMethod[] methods = {
            FeatureSelector.SelectionMethod.INFORMATION_GAIN,
            FeatureSelector.SelectionMethod.CHI_SQUARE,
            FeatureSelector.SelectionMethod.CORRELATION_BASED,
            FeatureSelector.SelectionMethod.HYBRID
        };
        
        System.out.println("Feature importance analysis:");
        for (FeatureSelector.SelectionMethod method : methods) {
            featureSelector.calculateFeatureImportance(method);
            double[] scores = featureSelector.getFeatureImportanceScores();
            
            System.out.printf("%s scores: ", method);
            for (int i = 0; i < scores.length; i++) {
                System.out.printf("F%d=%.3f ", i, scores[i]);
            }
            System.out.println();
        }
        
        // Select features using hybrid method
        int maxFeatures = Math.min(DataEnhanced.getNumOfNumericalFeatures(), 
                                 Math.max(2, DataEnhanced.getNumOfNumericalFeatures() * 2 / 3));
        
        selectedFeatures = featureSelector.selectFeatures(
            FeatureSelector.SelectionMethod.HYBRID, maxFeatures);
        
        // Remove redundant features
        selectedFeatures = featureSelector.removeRedundantFeatures(0.8);
        
        System.out.printf("Selected %d features: ", featureSelector.getNumSelectedFeatures());
        for (int i = 0; i < selectedFeatures.length; i++) {
            if (selectedFeatures[i]) {
                System.out.printf("F%d ", i);
            }
        }
        System.out.println();
    }
    
    private void performOptimization() {
        System.out.println("Phase 4: Enhanced ABC optimization...");
        
        abc = new EnhancedABC(config);
        abc.initialize(trainSet, cutPoints);
        
        System.out.printf("Initialized ABC with colony size: %d, max epochs: %d%n", 
                         config.getColonySize(), config.getMaxEpochs());
        
        // Optimization loop
        boolean done = false;
        while (!done) {
            done = abc.optimize();
            
            if (abc.getCurrentEpoch() % 100 == 0) {
                System.out.printf("Epoch %d: Best fitness = %.4f, Accuracy = %.4f%n", 
                                 abc.getCurrentEpoch(), abc.getBestFitness(),
                                 abc.getGlobalBest().getAccuracy());
            }
        }
        
        System.out.printf("Optimization completed after %d epochs%n", abc.getCurrentEpoch());
        System.out.printf("Final best fitness: %.4f%n", abc.getBestFitness());
        System.out.printf("Final accuracy: %.4f%n", abc.getGlobalBest().getAccuracy());
    }
    
    private void extractAndRefineRules() {
        System.out.println("Phase 5: Rule extraction and refinement...");
        
        // Get the best solution
        EnhancedHoney bestSolution = abc.getGlobalBest();
        
        // Extract rules from the best solution
        finalRules = extractRulesFromSolution(bestSolution);
        
        System.out.printf("Extracted %d rules%n", finalRules.size());
        
        // Calculate rule quality metrics
        for (EnhancedArea rule : finalRules) {
            rule.calculateRuleQualityMetrics(trainSet);
        }
        
        // Rule pruning
        List<EnhancedArea> prunedRules = new ArrayList<>();
        for (EnhancedArea rule : finalRules) {
            if (!rule.shouldBePruned(0.01, 0.5, 3.841, 2)) {
                prunedRules.add(rule);
            }
        }
        
        System.out.printf("After pruning: %d rules remain%n", prunedRules.size());
        finalRules = prunedRules;
        
        // Display rules
        System.out.println("Final rules:");
        for (int i = 0; i < finalRules.size(); i++) {
            EnhancedArea rule = finalRules.get(i);
            System.out.printf("Rule %d: %s%n", i + 1, rule.toString());
        }
    }
    
    private List<EnhancedArea> extractRulesFromSolution(EnhancedHoney solution) {
        List<EnhancedArea> rules = new ArrayList<>();
        
        // Get selected features from solution
        List<Integer> solutionFeatures = new ArrayList<>();
        double[] solutionCutPoints = new double[DataEnhanced.getNumOfNumericalFeatures()];
        
        for (int i = 0; i < DataEnhanced.getNumOfNumericalFeatures(); i++) {
            int nectarValue = solution.getNumNectar(i);
            if (nectarValue != 0 && nectarValue != cutPoints[i].length - 1) {
                solutionFeatures.add(i);
                solutionCutPoints[i] = cutPoints[i][nectarValue];
            }
        }
        
        if (solutionFeatures.isEmpty()) {
            return rules; // No valid rules
        }
        
        // Generate patterns and create rules
        Map<String, Map<Integer, Integer>> patterns = new HashMap<>();
        
        for (int i = 0; i < trainSet.getDataSetSize(); i++) {
            DataEnhanced instance = trainSet.getData(i);
            StringBuilder pattern = new StringBuilder();
            
            for (int featureIndex : solutionFeatures) {
                double value = instance.numericalFeatures[featureIndex];
                char bit = value <= solutionCutPoints[featureIndex] ? '0' : '1';
                pattern.append(bit);
            }
            
            String patternStr = pattern.toString();
            int label = instance.label;
            
            patterns.computeIfAbsent(patternStr, k -> new HashMap<>())
                   .put(label, patterns.get(patternStr).getOrDefault(label, 0) + 1);
        }
        
        // Create rules from patterns
        for (Map.Entry<String, Map<Integer, Integer>> entry : patterns.entrySet()) {
            String pattern = entry.getKey();
            Map<Integer, Integer> labelCounts = entry.getValue();
            
            // Find dominant class
            int maxCount = 0;
            int dominantLabel = -1;
            int totalCount = 0;
            
            for (Map.Entry<Integer, Integer> labelEntry : labelCounts.entrySet()) {
                int count = labelEntry.getValue();
                totalCount += count;
                if (count > maxCount) {
                    maxCount = count;
                    dominantLabel = labelEntry.getKey();
                }
            }
            
            if (dominantLabel != -1 && totalCount > 0) {
                EnhancedArea rule = new EnhancedArea();
                rule.setAreaCode(pattern);
                rule.setLabel(dominantLabel);
                rule.setSupport((double) totalCount / trainSet.getDataSetSize());
                rule.setConfidence((double) maxCount / totalCount);
                rule.setSelectedFeatures(new ArrayList<>(solutionFeatures));
                
                // Extract cut points for selected features
                double[] ruleCutPoints = new double[solutionFeatures.size()];
                for (int i = 0; i < solutionFeatures.size(); i++) {
                    ruleCutPoints[i] = solutionCutPoints[solutionFeatures.get(i)];
                }
                rule.setCutPoints(ruleCutPoints);
                
                rules.add(rule);
            }
        }
        
        return rules;
    }
    
    private void evaluatePerformance() {
        System.out.println("Phase 6: Comprehensive evaluation...");
        
        if (testSet == null || finalRules.isEmpty()) {
            System.out.println("Cannot evaluate: missing test set or rules");
            return;
        }
        
        // Evaluate on test set
        int correct = 0;
        int total = testSet.getDataSetSize();
        int covered = 0;
        
        int[][] confusionMatrix = new int[DataEnhanced.getLabelInterval()][DataEnhanced.getLabelInterval()];
        
        for (int i = 0; i < total; i++) {
            DataEnhanced instance = testSet.getData(i);
            int predicted = classifyInstance(instance);
            
            if (predicted != -1) {
                covered++;
                if (predicted == instance.label) {
                    correct++;
                }
                confusionMatrix[instance.label - 1][predicted - 1]++;
            } else {
                // Instance not covered by any rule - assign to most frequent class
                int mostFrequentClass = 1; // Simplified
                confusionMatrix[instance.label - 1][mostFrequentClass - 1]++;
            }
        }
        
        // Calculate metrics
        double accuracy = (double) correct / total;
        double coverage = (double) covered / total;
        double avgRuleLength = finalRules.stream().mapToInt(EnhancedArea::getRuleLength).average().orElse(0.0);
        
        // Store performance metrics
        performanceMetrics.put("accuracy", accuracy);
        performanceMetrics.put("coverage", coverage);
        performanceMetrics.put("numRules", finalRules.size());
        performanceMetrics.put("avgRuleLength", avgRuleLength);
        
        // Display results
        System.out.println("=== PERFORMANCE RESULTS ===");
        System.out.printf("Accuracy: %.4f (%.2f%%)%n", accuracy, accuracy * 100);
        System.out.printf("Coverage: %.4f (%.2f%%)%n", coverage, coverage * 100);
        System.out.printf("Number of rules: %d%n", finalRules.size());
        System.out.printf("Average rule length: %.2f%n", avgRuleLength);
        
        System.out.println("\nConfusion Matrix:");
        System.out.print("Actual\\Predicted\t");
        for (int i = 1; i <= DataEnhanced.getLabelInterval(); i++) {
            System.out.printf("%d\t", i);
        }
        System.out.println();
        
        for (int i = 0; i < DataEnhanced.getLabelInterval(); i++) {
            System.out.printf("Class %d\t\t", i + 1);
            for (int j = 0; j < DataEnhanced.getLabelInterval(); j++) {
                System.out.printf("%d\t", confusionMatrix[i][j]);
            }
            System.out.println();
        }
        
        // Rule quality analysis
        System.out.println("\nRule Quality Analysis:");
        for (int i = 0; i < finalRules.size(); i++) {
            EnhancedArea rule = finalRules.get(i);
            System.out.printf("Rule %d: Support=%.3f, Confidence=%.3f, Lift=%.3f, Significant=%s%n", 
                            i + 1, rule.getSupport(), rule.getConfidence(), 
                            rule.getLift(), rule.isSignificant() ? "Yes" : "No");
        }
    }
    
    private int classifyInstance(DataEnhanced instance) {
        double bestConfidence = 0.0;
        int bestLabel = -1;
        
        for (EnhancedArea rule : finalRules) {
            if (instanceMatchesRule(instance, rule)) {
                if (rule.getConfidence() > bestConfidence) {
                    bestConfidence = rule.getConfidence();
                    bestLabel = rule.getLabel();
                }
            }
        }
        
        return bestLabel;
    }
    
    private boolean instanceMatchesRule(DataEnhanced instance, EnhancedArea rule) {
        String areaCode = rule.getAreaCode();
        List<Integer> selectedFeatures = rule.getSelectedFeatures();
        double[] cutPoints = rule.getCutPoints();
        
        if (areaCode.length() != selectedFeatures.size()) {
            return false;
        }
        
        for (int i = 0; i < selectedFeatures.size(); i++) {
            int featureIndex = selectedFeatures.get(i);
            double value = instance.numericalFeatures[featureIndex];
            char expectedBit = areaCode.charAt(i);
            char actualBit = value <= cutPoints[i] ? '0' : '1';
            
            if (expectedBit != actualBit) {
                return false;
            }
        }
        
        return true;
    }
    
    // Getters for results
    public Map<String, Object> getPerformanceMetrics() {
        return new HashMap<>(performanceMetrics);
    }
    
    public List<EnhancedArea> getFinalRules() {
        return new ArrayList<>(finalRules);
    }
    
    public double[][] getCutPoints() {
        return cutPoints;
    }
    
    public boolean[] getSelectedFeatures() {
        return selectedFeatures;
    }
}