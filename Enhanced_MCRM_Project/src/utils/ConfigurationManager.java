package utils;

import java.util.HashMap;
import java.util.Map;

/**
 * Configuration Manager for Enhanced MCRM Algorithm
 */
public class ConfigurationManager {
    private Map<String, Object> parameters;
    
    // Default parameter values
    public static final int DEFAULT_COLONY_SIZE = 30;
    public static final int DEFAULT_MAX_EPOCHS = 1000;
    public static final int DEFAULT_LIMIT = 100;
    public static final double DEFAULT_FEATURE_SELECTION_PROB = 0.1;
    public static final double DEFAULT_MUTATION_RATE = 0.1;
    public static final double DEFAULT_CROSSOVER_RATE = 0.8;
    public static final int DEFAULT_POPULATION_SIZE = 100;
    public static final double DEFAULT_DIVERSITY_THRESHOLD = 0.1;
    public static final int DEFAULT_EARLY_STOPPING_PATIENCE = 50;
    
    // Discretization parameters
    public static final int DEFAULT_MAX_INTERVALS = 10;
    public static final int DEFAULT_MIN_INTERVALS = 2;
    public static final double DEFAULT_CHI_SQUARE_THRESHOLD = 3.841; // p=0.05, df=1
    public static final double DEFAULT_MDL_ALPHA = 0.01;
    
    // Multi-objective parameters
    public static final int DEFAULT_ARCHIVE_SIZE = 50;
    public static final double DEFAULT_CROWDING_DISTANCE_EPSILON = 1e-10;
    
    public ConfigurationManager() {
        this.parameters = new HashMap<>();
        initializeDefaults();
    }
    
    private void initializeDefaults() {
        // ABC Parameters
        parameters.put("colonySize", DEFAULT_COLONY_SIZE);
        parameters.put("maxEpochs", DEFAULT_MAX_EPOCHS);
        parameters.put("limit", DEFAULT_LIMIT);
        parameters.put("featureSelectionProbability", DEFAULT_FEATURE_SELECTION_PROB);
        
        // Genetic Algorithm Parameters
        parameters.put("mutationRate", DEFAULT_MUTATION_RATE);
        parameters.put("crossoverRate", DEFAULT_CROSSOVER_RATE);
        parameters.put("populationSize", DEFAULT_POPULATION_SIZE);
        
        // Adaptive Parameters
        parameters.put("diversityThreshold", DEFAULT_DIVERSITY_THRESHOLD);
        parameters.put("earlyStoppingPatience", DEFAULT_EARLY_STOPPING_PATIENCE);
        
        // Discretization Parameters
        parameters.put("maxIntervals", DEFAULT_MAX_INTERVALS);
        parameters.put("minIntervals", DEFAULT_MIN_INTERVALS);
        parameters.put("chiSquareThreshold", DEFAULT_CHI_SQUARE_THRESHOLD);
        parameters.put("mdlAlpha", DEFAULT_MDL_ALPHA);
        
        // Multi-objective Parameters
        parameters.put("archiveSize", DEFAULT_ARCHIVE_SIZE);
        parameters.put("crowdingDistanceEpsilon", DEFAULT_CROWDING_DISTANCE_EPSILON);
        
        // Performance Optimization Parameters
        parameters.put("enableCaching", true);
        parameters.put("enableParallelProcessing", false);
        parameters.put("cacheSize", 1000);
    }
    
    // Generic parameter getters and setters
    public void setParameter(String key, Object value) {
        parameters.put(key, value);
    }
    
    @SuppressWarnings("unchecked")
    public <T> T getParameter(String key, Class<T> type) {
        Object value = parameters.get(key);
        if (value != null && type.isAssignableFrom(value.getClass())) {
            return (T) value;
        }
        return null;
    }
    
    public Integer getIntParameter(String key) {
        return getParameter(key, Integer.class);
    }
    
    public Double getDoubleParameter(String key) {
        return getParameter(key, Double.class);
    }
    
    public Boolean getBooleanParameter(String key) {
        return getParameter(key, Boolean.class);
    }
    
    // Specific parameter getters for commonly used values
    public int getColonySize() {
        return getIntParameter("colonySize");
    }
    
    public int getMaxEpochs() {
        return getIntParameter("maxEpochs");
    }
    
    public int getLimit() {
        return getIntParameter("limit");
    }
    
    public double getFeatureSelectionProbability() {
        return getDoubleParameter("featureSelectionProbability");
    }
    
    public double getMutationRate() {
        return getDoubleParameter("mutationRate");
    }
    
    public double getCrossoverRate() {
        return getDoubleParameter("crossoverRate");
    }
    
    public int getPopulationSize() {
        return getIntParameter("populationSize");
    }
    
    public double getDiversityThreshold() {
        return getDoubleParameter("diversityThreshold");
    }
    
    public int getEarlyStoppingPatience() {
        return getIntParameter("earlyStoppingPatience");
    }
    
    public int getMaxIntervals() {
        return getIntParameter("maxIntervals");
    }
    
    public int getMinIntervals() {
        return getIntParameter("minIntervals");
    }
    
    public double getChiSquareThreshold() {
        return getDoubleParameter("chiSquareThreshold");
    }
    
    public double getMdlAlpha() {
        return getDoubleParameter("mdlAlpha");
    }
    
    public int getArchiveSize() {
        return getIntParameter("archiveSize");
    }
    
    public double getCrowdingDistanceEpsilon() {
        return getDoubleParameter("crowdingDistanceEpsilon");
    }
    
    public boolean isEnableCaching() {
        return getBooleanParameter("enableCaching");
    }
    
    public boolean isEnableParallelProcessing() {
        return getBooleanParameter("enableParallelProcessing");
    }
    
    public int getCacheSize() {
        return getIntParameter("cacheSize");
    }
    
    /**
     * Load configuration from properties file (simplified version)
     */
    public void loadFromProperties(String filePath) {
        // Implementation would load from properties file
        // For now, using defaults
    }
    
    /**
     * Save current configuration to properties file
     */
    public void saveToProperties(String filePath) {
        // Implementation would save to properties file
        // For now, just print current parameters
        System.out.println("Current Configuration:");
        for (Map.Entry<String, Object> entry : parameters.entrySet()) {
            System.out.println(entry.getKey() + " = " + entry.getValue());
        }
    }
    
    /**
     * Adaptive parameter adjustment based on algorithm performance
     */
    public void adaptParameters(double currentAccuracy, double bestAccuracy, 
                               double diversity, int stagnationCount) {
        // Adaptive mutation rate
        if (stagnationCount > getEarlyStoppingPatience() / 2) {
            double currentMutation = getMutationRate();
            setParameter("mutationRate", Math.min(0.3, currentMutation * 1.1));
        }
        
        // Adaptive colony size based on diversity
        if (diversity < getDiversityThreshold()) {
            int currentSize = getColonySize();
            setParameter("colonySize", Math.min(50, currentSize + 5));
        }
        
        // Adaptive feature selection probability
        if (currentAccuracy < bestAccuracy * 0.9) {
            double currentProb = getFeatureSelectionProbability();
            setParameter("featureSelectionProbability", Math.min(0.3, currentProb * 1.2));
        }
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("ConfigurationManager{\n");
        for (Map.Entry<String, Object> entry : parameters.entrySet()) {
            sb.append("  ").append(entry.getKey()).append(" = ").append(entry.getValue()).append("\n");
        }
        sb.append("}");
        return sb.toString();
    }
}