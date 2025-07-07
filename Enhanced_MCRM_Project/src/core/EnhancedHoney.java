package core;

import utils.*;
import java.util.*;

/**
 * Enhanced solution representation (equivalent to enhanced Honey class)
 */
public class EnhancedHoney implements Comparable<EnhancedHoney> {
    private int[] numNectar; // Solution representation
    private int trials;
    private double cost;
    private double fitness;
    private double selectionProbability;
    private double[][] cutPoints;
    private Random rand = new Random();
    
    // Multi-objective fitness values
    private double accuracy;
    private double complexity;
    private double coverage;
    private double[] objectives; // [accuracy, 1/complexity, coverage]
    
    // Rule quality metrics
    private double confidence;
    private double support;
    private double lift;
    private double conviction;
    
    // Performance optimization
    private Map<String, Object> cachedResults;
    private boolean isEvaluated;
    
    public EnhancedHoney(double[][] cutPoints) {
        if (DataEnhanced.getNumOfNumericalFeatures() != 0) {
            this.numNectar = new int[DataEnhanced.getNumOfNumericalFeatures()];
        }
        
        this.cost = 0;
        this.trials = 0;
        this.fitness = 0.0;
        this.selectionProbability = 0.0;
        this.cutPoints = cutPoints;
        this.objectives = new double[3];
        this.cachedResults = new HashMap<>();
        this.isEvaluated = false;
        
        initNectar();
    }
    
    private void initNectar() {
        if (numNectar != null) {
            for (int i = 0; i < numNectar.length; i++) {
                numNectar[i] = rand.nextInt(cutPoints[i].length);
            }
        }
    }
    
    /**
     * Enhanced fitness calculation with multiple objectives
     */
    public void computeEnhancedCost(DataSetEnhanced trainSet) {
        if (isEvaluated) return; // Use cache
        
        DataSetEnhanced data = trainSet;
        
        // Extract cut points and selected features
        double[] extractedCutPoints = extractCutPoints(this.numNectar);
        List<Integer> selectedFeatures = getSelectedFeatures();
        
        if (selectedFeatures.isEmpty()) {
            this.cost = Double.MAX_VALUE;
            this.accuracy = 0.0;
            this.complexity = Double.MAX_VALUE;
            this.coverage = 0.0;
            return;
        }
        
        // Generate rules and calculate metrics
        List<EnhancedArea> rules = generateRules(data, extractedCutPoints, selectedFeatures);
        
        // Calculate accuracy
        this.accuracy = calculateAccuracy(data, rules);
        
        // Calculate complexity (inverse of number of conditions)
        this.complexity = calculateComplexity(rules);
        
        // Calculate coverage
        this.coverage = calculateCoverage(data, rules);
        
        // Calculate additional rule quality metrics
        calculateRuleQualityMetrics(data, rules);
        
        // Multi-objective fitness
        this.objectives[0] = this.accuracy;
        this.objectives[1] = 1.0 / (1.0 + this.complexity); // Minimize complexity
        this.objectives[2] = this.coverage;
        
        // Combined fitness (weighted sum)
        this.cost = -(0.5 * accuracy + 0.3 * objectives[1] + 0.2 * coverage);
        this.fitness = 1.0 / (1.0 + Math.abs(this.cost));
        
        this.isEvaluated = true;
    }
    
    /**
     * Generate enhanced rules with quality assessment
     */
    private List<EnhancedArea> generateRules(DataSetEnhanced data, double[] cutPoints, List<Integer> selectedFeatures) {
        List<EnhancedArea> rules = new ArrayList<>();
        Map<String, Map<Integer, Integer>> patterns = new HashMap<>();
        
        // Generate patterns
        for (int i = 0; i < data.getDataSetSize(); i++) {
            StringBuilder pattern = new StringBuilder();
            for (int featureIndex : selectedFeatures) {
                double value = data.getData(i).numericalFeatures[featureIndex];
                char bit = value <= cutPoints[featureIndex] ? '0' : '1';
                pattern.append(bit);
            }
            
            String patternStr = pattern.toString();
            int label = data.getData(i).label;
            
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
                rule.setSupport((double) totalCount / data.getDataSetSize());
                rule.setConfidence((double) maxCount / totalCount);
                rule.setSelectedFeatures(new ArrayList<>(selectedFeatures));
                rule.setCutPoints(extractCutPointsForFeatures(selectedFeatures, cutPoints));
                
                rules.add(rule);
            }
        }
        
        return rules;
    }
    
    private double calculateAccuracy(DataSetEnhanced data, List<EnhancedArea> rules) {
        int correct = 0;
        int total = data.getDataSetSize();
        
        for (int i = 0; i < total; i++) {
            DataEnhanced instance = data.getData(i);
            int predictedLabel = classifyInstance(instance, rules);
            
            if (predictedLabel == instance.label) {
                correct++;
            }
        }
        
        return (double) correct / total;
    }
    
    private double calculateComplexity(List<EnhancedArea> rules) {
        double totalComplexity = 0.0;
        
        for (EnhancedArea rule : rules) {
            // Complexity based on number of conditions
            totalComplexity += rule.getSelectedFeatures().size();
        }
        
        return totalComplexity / rules.size();
    }
    
    private double calculateCoverage(DataSetEnhanced data, List<EnhancedArea> rules) {
        int covered = 0;
        int total = data.getDataSetSize();
        
        for (int i = 0; i < total; i++) {
            DataEnhanced instance = data.getData(i);
            if (isInstanceCovered(instance, rules)) {
                covered++;
            }
        }
        
        return (double) covered / total;
    }
    
    private void calculateRuleQualityMetrics(DataSetEnhanced data, List<EnhancedArea> rules) {
        double totalSupport = 0.0;
        double totalConfidence = 0.0;
        double totalLift = 0.0;
        
        for (EnhancedArea rule : rules) {
            totalSupport += rule.getSupport();
            totalConfidence += rule.getConfidence();
            // Lift calculation would require class prior probabilities
            totalLift += 1.0; // Simplified
        }
        
        int numRules = rules.size();
        this.support = numRules > 0 ? totalSupport / numRules : 0.0;
        this.confidence = numRules > 0 ? totalConfidence / numRules : 0.0;
        this.lift = numRules > 0 ? totalLift / numRules : 0.0;
    }
    
    private int classifyInstance(DataEnhanced instance, List<EnhancedArea> rules) {
        // Find the best matching rule
        double bestConfidence = 0.0;
        int bestLabel = -1;
        
        for (EnhancedArea rule : rules) {
            if (doesInstanceMatchRule(instance, rule)) {
                if (rule.getConfidence() > bestConfidence) {
                    bestConfidence = rule.getConfidence();
                    bestLabel = rule.getLabel();
                }
            }
        }
        
        return bestLabel;
    }
    
    private boolean isInstanceCovered(DataEnhanced instance, List<EnhancedArea> rules) {
        for (EnhancedArea rule : rules) {
            if (doesInstanceMatchRule(instance, rule)) {
                return true;
            }
        }
        return false;
    }
    
    private boolean doesInstanceMatchRule(DataEnhanced instance, EnhancedArea rule) {
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
    
    private List<Integer> getSelectedFeatures() {
        List<Integer> selected = new ArrayList<>();
        
        for (int i = 0; i < DataEnhanced.getNumOfNumericalFeatures(); i++) {
            if (cutPoints[i].length - 1 != numNectar[i] && numNectar[i] != 0) {
                selected.add(i);
            }
        }
        
        return selected;
    }
    
    private double[] extractCutPoints(int[] nectarValues) {
        double[] extracted = new double[DataEnhanced.getNumOfNumericalFeatures()];
        
        for (int i = 0; i < nectarValues.length; i++) {
            if (nectarValues[i] < cutPoints[i].length) {
                extracted[i] = cutPoints[i][nectarValues[i]];
            } else {
                extracted[i] = cutPoints[i][cutPoints[i].length - 1];
            }
        }
        
        return extracted;
    }
    
    private double[] extractCutPointsForFeatures(List<Integer> features, double[] cutPoints) {
        double[] extracted = new double[features.size()];
        
        for (int i = 0; i < features.size(); i++) {
            extracted[i] = cutPoints[features.get(i)];
        }
        
        return extracted;
    }
    
    /**
     * Dominance relation for multi-objective optimization
     */
    public boolean dominates(EnhancedHoney other) {
        boolean atLeastOneBetter = false;
        
        for (int i = 0; i < objectives.length; i++) {
            if (this.objectives[i] < other.objectives[i]) {
                return false; // This solution is worse in at least one objective
            }
            if (this.objectives[i] > other.objectives[i]) {
                atLeastOneBetter = true;
            }
        }
        
        return atLeastOneBetter;
    }
    
    /**
     * Calculate crowding distance for NSGA-II
     */
    public double calculateCrowdingDistance(List<EnhancedHoney> front) {
        if (front.size() <= 2) return Double.MAX_VALUE;
        
        double distance = 0.0;
        
        for (int obj = 0; obj < objectives.length; obj++) {
            // Sort by objective
            final int objectiveIndex = obj;
            front.sort((a, b) -> Double.compare(a.objectives[objectiveIndex], b.objectives[objectiveIndex]));
            
            // Find min and max
            double minObj = front.get(0).objectives[obj];
            double maxObj = front.get(front.size() - 1).objectives[obj];
            double range = maxObj - minObj;
            
            if (range == 0) continue;
            
            // Find this solution's position
            int index = front.indexOf(this);
            if (index == 0 || index == front.size() - 1) {
                return Double.MAX_VALUE; // Boundary solutions
            }
            
            // Calculate distance
            double prevObj = front.get(index - 1).objectives[obj];
            double nextObj = front.get(index + 1).objectives[obj];
            distance += (nextObj - prevObj) / range;
        }
        
        return distance;
    }
    
    @Override
    public int compareTo(EnhancedHoney other) {
        return Double.compare(other.fitness, this.fitness); // Descending order
    }
    
    /**
     * Create a deep copy of this solution
     */
    public EnhancedHoney deepCopy() {
        EnhancedHoney copy = new EnhancedHoney(this.cutPoints);
        
        if (this.numNectar != null) {
            copy.numNectar = this.numNectar.clone();
        }
        
        copy.trials = this.trials;
        copy.cost = this.cost;
        copy.fitness = this.fitness;
        copy.accuracy = this.accuracy;
        copy.complexity = this.complexity;
        copy.coverage = this.coverage;
        copy.objectives = this.objectives.clone();
        copy.isEvaluated = this.isEvaluated;
        
        return copy;
    }
    
    // Getters and setters
    public int[] getNumNectar() { return numNectar; }
    public void setNumNectar(int[] numNectar) { this.numNectar = numNectar; }
    
    public int getNumNectar(int index) { return numNectar[index]; }
    public void setNumNectar(int index, int value) { 
        this.numNectar[index] = value; 
        this.isEvaluated = false; // Invalidate cache
    }
    
    public double getCost() { return cost; }
    public void setCost(double cost) { this.cost = cost; }
    
    public double getFitness() { return fitness; }
    public void setFitness(double fitness) { this.fitness = fitness; }
    
    public int getTrials() { return trials; }
    public void setTrials(int trials) { this.trials = trials; }
    
    public double getSelectionProbability() { return selectionProbability; }
    public void setSelectionProbability(double probability) { this.selectionProbability = probability; }
    
    public double[] getObjectives() { return objectives.clone(); }
    public double getAccuracy() { return accuracy; }
    public double getComplexity() { return complexity; }
    public double getCoverage() { return coverage; }
    public double getConfidence() { return confidence; }
    public double getSupport() { return support; }
    public double getLift() { return lift; }
}