package core;

import java.util.*;

/**
 * Enhanced rule representation (equivalent to enhanced Area class)
 */
public class EnhancedArea {
    private List<Integer> selectedFeatures;
    private double[] cutPoints;
    private boolean remove = false;
    private String areaCode;
    private int label;
    private int[] labelFrequency;
    
    // Enhanced rule quality metrics
    private double support;
    private double confidence;
    private double lift;
    private double conviction;
    private double leverage;
    private double coverage;
    private int truePositives;
    private int falsePositives;
    private int falseNegatives;
    private int trueNegatives;
    
    // Rule complexity metrics
    private int ruleLength;
    private double specificity;
    private double generality;
    
    // Statistical significance
    private double chiSquareStatistic;
    private double pValue;
    private boolean isSignificant;
    
    public EnhancedArea() {
        this.selectedFeatures = new ArrayList<>();
        this.labelFrequency = new int[utils.DataEnhanced.getLabelInterval()];
        this.ruleLength = 0;
        this.isSignificant = false;
    }
    
    /**
     * Calculate comprehensive rule quality metrics
     */
    public void calculateRuleQualityMetrics(utils.DataSetEnhanced dataset) {
        calculateBasicMetrics(dataset);
        calculateAdvancedMetrics(dataset);
        calculateStatisticalSignificance(dataset);
    }
    
    private void calculateBasicMetrics(utils.DataSetEnhanced dataset) {
        int totalInstances = dataset.getDataSetSize();
        int ruleMatches = 0;
        int correctPredictions = 0;
        int classTotalInstances = 0;
        
        // Reset counters
        truePositives = 0;
        falsePositives = 0;
        falseNegatives = 0;
        trueNegatives = 0;
        
        for (int i = 0; i < totalInstances; i++) {
            utils.DataEnhanced instance = dataset.getData(i);
            boolean instanceMatchesRule = matchesRule(instance);
            boolean instanceHasTargetClass = (instance.label == this.label);
            
            if (instanceHasTargetClass) {
                classTotalInstances++;
                if (instanceMatchesRule) {
                    truePositives++;
                    correctPredictions++;
                } else {
                    falseNegatives++;
                }
            } else {
                if (instanceMatchesRule) {
                    falsePositives++;
                } else {
                    trueNegatives++;
                }
            }
            
            if (instanceMatchesRule) {
                ruleMatches++;
            }
        }
        
        // Calculate basic metrics
        this.support = (double) ruleMatches / totalInstances;
        this.confidence = ruleMatches > 0 ? (double) correctPredictions / ruleMatches : 0.0;
        this.coverage = (double) ruleMatches / totalInstances;
        
        // Calculate precision, recall, specificity
        double precision = (truePositives + falsePositives) > 0 ? 
                          (double) truePositives / (truePositives + falsePositives) : 0.0;
        double recall = (truePositives + falseNegatives) > 0 ? 
                       (double) truePositives / (truePositives + falseNegatives) : 0.0;
        this.specificity = (trueNegatives + falsePositives) > 0 ? 
                          (double) trueNegatives / (trueNegatives + falsePositives) : 0.0;
        
        // Calculate rule length and generality
        this.ruleLength = selectedFeatures.size();
        this.generality = this.coverage; // Simplified - could be more sophisticated
    }
    
    private void calculateAdvancedMetrics(utils.DataSetEnhanced dataset) {
        int totalInstances = dataset.getDataSetSize();
        
        // Calculate class prior probability
        int classTotalInstances = 0;
        for (int i = 0; i < totalInstances; i++) {
            if (dataset.getData(i).label == this.label) {
                classTotalInstances++;
            }
        }
        double classPrior = (double) classTotalInstances / totalInstances;
        
        // Calculate lift
        this.lift = classPrior > 0 ? this.confidence / classPrior : 0.0;
        
        // Calculate conviction
        if (this.confidence < 1.0) {
            this.conviction = (1.0 - classPrior) / (1.0 - this.confidence);
        } else {
            this.conviction = Double.MAX_VALUE;
        }
        
        // Calculate leverage
        this.leverage = this.support - (this.coverage * classPrior);
    }
    
    private void calculateStatisticalSignificance(utils.DataSetEnhanced dataset) {
        // Chi-square test for rule significance
        int totalInstances = dataset.getDataSetSize();
        
        // Expected frequencies
        double expectedTP = (double) (truePositives + falsePositives) * 
                           (truePositives + falseNegatives) / totalInstances;
        double expectedFP = (double) (truePositives + falsePositives) * 
                           (falsePositives + trueNegatives) / totalInstances;
        double expectedFN = (double) (falseNegatives + trueNegatives) * 
                           (truePositives + falseNegatives) / totalInstances;
        double expectedTN = (double) (falseNegatives + trueNegatives) * 
                           (falsePositives + trueNegatives) / totalInstances;
        
        // Chi-square statistic
        this.chiSquareStatistic = 0.0;
        if (expectedTP > 0) {
            this.chiSquareStatistic += Math.pow(truePositives - expectedTP, 2) / expectedTP;
        }
        if (expectedFP > 0) {
            this.chiSquareStatistic += Math.pow(falsePositives - expectedFP, 2) / expectedFP;
        }
        if (expectedFN > 0) {
            this.chiSquareStatistic += Math.pow(falseNegatives - expectedFN, 2) / expectedFN;
        }
        if (expectedTN > 0) {
            this.chiSquareStatistic += Math.pow(trueNegatives - expectedTN, 2) / expectedTN;
        }
        
        // Simplified p-value calculation (critical value for p=0.05, df=1 is 3.841)
        this.isSignificant = this.chiSquareStatistic > 3.841;
        this.pValue = this.isSignificant ? 0.01 : 0.1; // Simplified
    }
    
    private boolean matchesRule(utils.DataEnhanced instance) {
        if (areaCode == null || selectedFeatures == null || cutPoints == null) {
            return false;
        }
        
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
    
    /**
     * Rule pruning with statistical criteria
     */
    public boolean shouldBePruned(double minSupport, double minConfidence, 
                                 double minSignificance, int minCoverage) {
        return this.support < minSupport || 
               this.confidence < minConfidence ||
               this.chiSquareStatistic < minSignificance ||
               (truePositives + falsePositives) < minCoverage;
    }
    
    /**
     * Rule simplification - remove redundant conditions
     */
    public EnhancedArea simplifyRule(utils.DataSetEnhanced dataset) {
        if (selectedFeatures.size() <= 1) {
            return this; // Cannot simplify further
        }
        
        EnhancedArea simplifiedRule = this.deepCopy();
        boolean improved = true;
        
        while (improved && simplifiedRule.selectedFeatures.size() > 1) {
            improved = false;
            double currentAccuracy = simplifiedRule.confidence;
            
            // Try removing each condition
            for (int i = simplifiedRule.selectedFeatures.size() - 1; i >= 0; i--) {
                EnhancedArea testRule = simplifiedRule.deepCopy();
                testRule.selectedFeatures.remove(i);
                
                // Update area code and cut points
                StringBuilder newAreaCode = new StringBuilder();
                List<Double> newCutPoints = new ArrayList<>();
                
                for (int j = 0; j < testRule.selectedFeatures.size(); j++) {
                    if (j != i) {
                        newAreaCode.append(testRule.areaCode.charAt(j));
                        newCutPoints.add(testRule.cutPoints[j]);
                    }
                }
                
                testRule.areaCode = newAreaCode.toString();
                testRule.cutPoints = newCutPoints.stream().mapToDouble(Double::doubleValue).toArray();
                
                // Evaluate simplified rule
                testRule.calculateRuleQualityMetrics(dataset);
                
                // If simplified rule is not worse, use it
                if (testRule.confidence >= currentAccuracy * 0.95 && // Allow 5% degradation
                    testRule.support >= simplifiedRule.support * 0.9) { // Allow 10% support reduction
                    
                    simplifiedRule = testRule;
                    improved = true;
                    break;
                }
            }
        }
        
        return simplifiedRule;
    }
    
    /**
     * Create a deep copy of this rule
     */
    public EnhancedArea deepCopy() {
        EnhancedArea copy = new EnhancedArea();
        
        copy.selectedFeatures = new ArrayList<>(this.selectedFeatures);
        if (this.cutPoints != null) {
            copy.cutPoints = this.cutPoints.clone();
        }
        copy.remove = this.remove;
        copy.areaCode = this.areaCode;
        copy.label = this.label;
        copy.labelFrequency = this.labelFrequency.clone();
        
        // Copy metrics
        copy.support = this.support;
        copy.confidence = this.confidence;
        copy.lift = this.lift;
        copy.conviction = this.conviction;
        copy.leverage = this.leverage;
        copy.coverage = this.coverage;
        copy.truePositives = this.truePositives;
        copy.falsePositives = this.falsePositives;
        copy.falseNegatives = this.falseNegatives;
        copy.trueNegatives = this.trueNegatives;
        copy.ruleLength = this.ruleLength;
        copy.specificity = this.specificity;
        copy.generality = this.generality;
        copy.chiSquareStatistic = this.chiSquareStatistic;
        copy.pValue = this.pValue;
        copy.isSignificant = this.isSignificant;
        
        return copy;
    }
    
    // Getters and setters
    public List<Integer> getSelectedFeatures() { return selectedFeatures; }
    public void setSelectedFeatures(List<Integer> selectedFeatures) { this.selectedFeatures = selectedFeatures; }
    
    public double[] getCutPoints() { return cutPoints; }
    public void setCutPoints(double[] cutPoints) { this.cutPoints = cutPoints; }
    
    public boolean isRemove() { return remove; }
    public void setRemove(boolean remove) { this.remove = remove; }
    
    public String getAreaCode() { return areaCode; }
    public void setAreaCode(String areaCode) { this.areaCode = areaCode; }
    
    public int getLabel() { return label; }
    public void setLabel(int label) { this.label = label; }
    
    public int[] getLabelFrequency() { return labelFrequency; }
    public void setLabelFrequency(int[] labelFrequency) { this.labelFrequency = labelFrequency; }
    
    // Quality metrics getters
    public double getSupport() { return support; }
    public void setSupport(double support) { this.support = support; }
    
    public double getConfidence() { return confidence; }
    public void setConfidence(double confidence) { this.confidence = confidence; }
    
    public double getLift() { return lift; }
    public double getConviction() { return conviction; }
    public double getLeverage() { return leverage; }
    public double getCoverage() { return coverage; }
    public int getRuleLength() { return ruleLength; }
    public double getSpecificity() { return specificity; }
    public double getGenerality() { return generality; }
    public double getChiSquareStatistic() { return chiSquareStatistic; }
    public double getPValue() { return pValue; }
    public boolean isSignificant() { return isSignificant; }
    
    public int getTruePositives() { return truePositives; }
    public int getFalsePositives() { return falsePositives; }
    public int getFalseNegatives() { return falseNegatives; }
    public int getTrueNegatives() { return trueNegatives; }
    
    @Override
    public String toString() {
        return String.format("Rule{pattern='%s', label=%d, support=%.3f, confidence=%.3f, lift=%.3f, length=%d}", 
                           areaCode, label, support, confidence, lift, ruleLength);
    }
}