package core;

import utils.*;
import java.util.*;

/**
 * Enhanced Artificial Bee Colony with multi-objective optimization and advanced search strategies
 */
public class EnhancedABC {
    // Basic ABC parameters
    private int MAX_LENGTH;
    private int NP; // Colony size
    private int FOOD_NUMBER; // Number of food sources
    private int LIMIT; // Abandonment limit
    private int MAX_EPOCH;
    
    // Enhanced parameters
    private ConfigurationManager config;
    private List<EnhancedHoney> foodSources;
    private List<EnhancedHoney> archive; // For multi-objective optimization
    private EnhancedHoney gBest;
    private double[][] cutPoints;
    private DataSetEnhanced trainSet;
    
    // Search strategy parameters
    private Random rand;
    private double featureSelectionProbability;
    private int currentEpoch;
    private int stagnationCount;
    private double bestFitnessHistory;
    
    // Multi-objective optimization
    private List<List<EnhancedHoney>> fronts;
    private boolean useMultiObjective;
    
    // Adaptive parameters
    private double diversityThreshold;
    private int adaptiveColonySize;
    private double adaptiveMutationRate;
    
    // Performance optimization
    private Map<String, Double> fitnessCache;
    private boolean enableCaching;
    
    public EnhancedABC(ConfigurationManager config) {
        this.config = config;
        this.rand = new Random();
        
        // Initialize parameters from config
        this.NP = config.getColonySize();
        this.FOOD_NUMBER = NP / 2;
        this.LIMIT = config.getLimit();
        this.MAX_EPOCH = config.getMaxEpochs();
        this.featureSelectionProbability = config.getFeatureSelectionProbability();
        this.diversityThreshold = config.getDiversityThreshold();
        this.enableCaching = config.isEnableCaching();
        
        // Initialize adaptive parameters
        this.adaptiveColonySize = this.NP;
        this.adaptiveMutationRate = config.getMutationRate();
        
        // Initialize collections
        this.foodSources = new ArrayList<>();
        this.archive = new ArrayList<>();
        this.fronts = new ArrayList<>();
        this.fitnessCache = new HashMap<>();
        
        this.currentEpoch = 0;
        this.stagnationCount = 0;
        this.bestFitnessHistory = Double.NEGATIVE_INFINITY;
        this.useMultiObjective = false; // Disable for simple testing
    }
    
    public void initialize(DataSetEnhanced trainSet, double[][] cutPoints) {
        this.trainSet = trainSet;
        this.cutPoints = cutPoints;
        this.MAX_LENGTH = cutPoints.length;
        
        // Initialize food sources
        foodSources.clear();
        for (int i = 0; i < FOOD_NUMBER; i++) {
            EnhancedHoney honey = new EnhancedHoney(cutPoints);
            initializeHoney(honey);
            foodSources.add(honey);
        }
        
        // Initial evaluation
        evaluatePopulation();
        
        // Initialize global best
        updateGlobalBest();
    }
    
    private void initializeHoney(EnhancedHoney honey) {
        int[] nectar = new int[MAX_LENGTH];
        
        // Multiple initialization strategies
        double strategy = rand.nextDouble();
        
        if (strategy < 0.25) {
            // Random initialization
            for (int i = 0; i < MAX_LENGTH; i++) {
                nectar[i] = rand.nextInt(cutPoints[i].length);
            }
        } else if (strategy < 0.5) {
            // Feature importance-based initialization
            initializeBasedOnFeatureImportance(nectar);
        } else if (strategy < 0.75) {
            // Entropy-based initialization
            initializeBasedOnEntropy(nectar);
        } else {
            // Hybrid initialization
            initializeHybrid(nectar);
        }
        
        honey.setNumNectar(nectar);
    }
    
    private void initializeBasedOnFeatureImportance(int[] nectar) {
        // Use feature selector to get importance scores
        feature_selection.FeatureSelector selector = new feature_selection.FeatureSelector(trainSet, config);
        selector.calculateFeatureImportance(feature_selection.FeatureSelector.SelectionMethod.INFORMATION_GAIN);
        double[] importanceScores = selector.getFeatureImportanceScores();
        
        for (int i = 0; i < MAX_LENGTH; i++) {
            if (importanceScores[i] > 0.5) {
                // High importance features - select middle cut points
                nectar[i] = cutPoints[i].length / 2;
            } else {
                // Low importance features - random or boundary values
                nectar[i] = rand.nextBoolean() ? 0 : cutPoints[i].length - 1;
            }
        }
    }
    
    private void initializeBasedOnEntropy(int[] nectar) {
        // Initialize based on entropy reduction
        for (int i = 0; i < MAX_LENGTH; i++) {
            // Select cut point that maximizes information gain
            double bestGain = -1.0;
            int bestCutPoint = 0;
            
            for (int cp = 0; cp < cutPoints[i].length; cp++) {
                double gain = calculateInformationGain(i, cp);
                if (gain > bestGain) {
                    bestGain = gain;
                    bestCutPoint = cp;
                }
            }
            
            nectar[i] = bestCutPoint;
        }
    }
    
    private void initializeHybrid(int[] nectar) {
        // Combine multiple strategies
        for (int i = 0; i < MAX_LENGTH; i++) {
            double strategy = rand.nextDouble();
            if (strategy < 0.5) {
                nectar[i] = rand.nextInt(cutPoints[i].length);
            } else {
                nectar[i] = cutPoints[i].length / 2; // Middle cut point
            }
        }
    }
    
    private double calculateInformationGain(int featureIndex, int cutPointIndex) {
        // Simplified information gain calculation
        if (cutPointIndex >= cutPoints[featureIndex].length) {
            return 0.0;
        }
        
        double cutPoint = cutPoints[featureIndex][cutPointIndex];
        
        // Calculate entropy before and after split
        Map<Integer, Integer> classCountBefore = new HashMap<>();
        Map<Integer, Integer> classCountLeft = new HashMap<>();
        Map<Integer, Integer> classCountRight = new HashMap<>();
        
        for (int i = 0; i < trainSet.getDataSetSize(); i++) {
            DataEnhanced data = trainSet.getData(i);
            int label = data.label;
            
            classCountBefore.put(label, classCountBefore.getOrDefault(label, 0) + 1);
            
            if (data.numericalFeatures[featureIndex] <= cutPoint) {
                classCountLeft.put(label, classCountLeft.getOrDefault(label, 0) + 1);
            } else {
                classCountRight.put(label, classCountRight.getOrDefault(label, 0) + 1);
            }
        }
        
        double entropyBefore = calculateEntropy(classCountBefore, trainSet.getDataSetSize());
        double entropyLeft = calculateEntropy(classCountLeft, classCountLeft.values().stream().mapToInt(Integer::intValue).sum());
        double entropyRight = calculateEntropy(classCountRight, classCountRight.values().stream().mapToInt(Integer::intValue).sum());
        
        int leftSize = classCountLeft.values().stream().mapToInt(Integer::intValue).sum();
        int rightSize = classCountRight.values().stream().mapToInt(Integer::intValue).sum();
        int totalSize = trainSet.getDataSetSize();
        
        double entropyAfter = ((double) leftSize / totalSize) * entropyLeft + 
                             ((double) rightSize / totalSize) * entropyRight;
        
        return entropyBefore - entropyAfter;
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
    
    public boolean optimize() {
        if (currentEpoch < MAX_EPOCH) {
            currentEpoch++;
            
            // Employed bee phase
            sendEmployedBees();
            
            // Calculate fitness and probabilities
            evaluatePopulation();
            calculateProbabilities();
            
            // Onlooker bee phase
            sendOnlookerBees();
            
            // Update global best
            updateGlobalBest();
            
            // Scout bee phase
            sendScoutBees();
            
            // Multi-objective operations
            if (useMultiObjective) {
                updateArchive();
                performNSGAIISelection();
            }
            
            // Adaptive parameter adjustment
            adaptParameters();
            
            // Check for convergence
            checkConvergence();
            
            return false; // Continue optimization
        }
        
        return true; // Optimization completed
    }
    
    private void sendEmployedBees() {
        for (int i = 0; i < FOOD_NUMBER; i++) {
            // Select neighbor
            int neighborIndex = getExclusiveRandomNumber(FOOD_NUMBER - 1, i);
            EnhancedHoney currentBee = foodSources.get(i);
            EnhancedHoney neighborBee = foodSources.get(neighborIndex);
            
            // Apply search strategy
            searchWithStrategy(currentBee, neighborBee);
        }
    }
    
    private void searchWithStrategy(EnhancedHoney currentBee, EnhancedHoney neighborBee) {
        double strategy = rand.nextDouble();
        
        if (strategy < 0.25) {
            // Large neighborhood search
            largeNeighborhoodSearch(currentBee, neighborBee);
        } else if (strategy < 0.5) {
            // Adaptive search
            adaptiveSearch(currentBee, neighborBee);
        } else if (strategy < 0.75) {
            // Local search
            localSearch(currentBee);
        } else {
            // Lévy flight
            levyFlightSearch(currentBee);
        }
    }
    
    private void largeNeighborhoodSearch(EnhancedHoney currentBee, EnhancedHoney neighborBee) {
        EnhancedHoney newBee = currentBee.deepCopy();
        int numChanges = 1 + rand.nextInt(3); // Change 1-3 parameters
        
        for (int i = 0; i < numChanges; i++) {
            int paramIndex = rand.nextInt(MAX_LENGTH);
            int currentValue = currentBee.getNumNectar(paramIndex);
            int neighborValue = neighborBee.getNumNectar(paramIndex);
            
            // Calculate new value using ABC formula with enhancement
            double phi = -1.0 + 2.0 * rand.nextDouble();
            int newValue = (int) (currentValue + phi * (currentValue - neighborValue));
            
            // Boundary handling
            newValue = Math.max(0, Math.min(cutPoints[paramIndex].length - 1, newValue));
            newBee.setNumNectar(paramIndex, newValue);
        }
        
        evaluateAndReplace(currentBee, newBee);
    }
    
    private void adaptiveSearch(EnhancedHoney currentBee, EnhancedHoney neighborBee) {
        EnhancedHoney newBee = currentBee.deepCopy();
        
        // Adaptive step size based on fitness difference
        double fitnessDiff = Math.abs(currentBee.getFitness() - neighborBee.getFitness());
        double stepSize = Math.min(1.0, Math.max(0.1, fitnessDiff));
        
        int paramIndex = rand.nextInt(MAX_LENGTH);
        int currentValue = currentBee.getNumNectar(paramIndex);
        int neighborValue = neighborBee.getNumNectar(paramIndex);
        
        double phi = -stepSize + 2.0 * stepSize * rand.nextDouble();
        int newValue = (int) (currentValue + phi * (currentValue - neighborValue));
        
        newValue = Math.max(0, Math.min(cutPoints[paramIndex].length - 1, newValue));
        newBee.setNumNectar(paramIndex, newValue);
        
        evaluateAndReplace(currentBee, newBee);
    }
    
    private void localSearch(EnhancedHoney currentBee) {
        EnhancedHoney newBee = currentBee.deepCopy();
        
        // Local perturbation
        int paramIndex = rand.nextInt(MAX_LENGTH);
        int currentValue = currentBee.getNumNectar(paramIndex);
        int direction = rand.nextBoolean() ? 1 : -1;
        int newValue = currentValue + direction;
        
        newValue = Math.max(0, Math.min(cutPoints[paramIndex].length - 1, newValue));
        newBee.setNumNectar(paramIndex, newValue);
        
        evaluateAndReplace(currentBee, newBee);
    }
    
    private void levyFlightSearch(EnhancedHoney currentBee) {
        EnhancedHoney newBee = currentBee.deepCopy();
        
        // Lévy flight step
        double levy = generateLevyFlight();
        int paramIndex = rand.nextInt(MAX_LENGTH);
        int currentValue = currentBee.getNumNectar(paramIndex);
        
        int step = (int) (levy * cutPoints[paramIndex].length / 4);
        int newValue = currentValue + step;
        
        newValue = Math.max(0, Math.min(cutPoints[paramIndex].length - 1, newValue));
        newBee.setNumNectar(paramIndex, newValue);
        
        evaluateAndReplace(currentBee, newBee);
    }
    
    private double generateLevyFlight() {
        // Simplified Lévy flight generation
        double beta = 1.5;
        double sigma = Math.pow(
            gamma(1 + beta) * Math.sin(Math.PI * beta / 2) / 
            (gamma((1 + beta) / 2) * beta * Math.pow(2, (beta - 1) / 2)), 
            1 / beta
        );
        
        double u = rand.nextGaussian() * sigma;
        double v = rand.nextGaussian();
        
        return u / Math.pow(Math.abs(v), 1 / beta);
    }
    
    private double gamma(double x) {
        // Simplified gamma function approximation
        return Math.sqrt(2 * Math.PI / x) * Math.pow(x / Math.E, x);
    }
    
    private void evaluateAndReplace(EnhancedHoney current, EnhancedHoney newSolution) {
        evaluateHoney(newSolution);
        
        if (newSolution.getFitness() > current.getFitness()) {
            // Replace with better solution
            int index = foodSources.indexOf(current);
            if (index >= 0) {
                foodSources.set(index, newSolution);
                current.setTrials(0); // Reset trials for successful update
            }
        } else {
            current.setTrials(current.getTrials() + 1);
        }
    }
    
    private void sendOnlookerBees() {
        int onlookerCount = 0;
        int index = 0;
        
        while (onlookerCount < FOOD_NUMBER && index < FOOD_NUMBER) {
            if (rand.nextDouble() < foodSources.get(index).getSelectionProbability()) {
                onlookerCount++;
                
                // Select neighbor for onlooker bee
                int neighborIndex = getExclusiveRandomNumber(FOOD_NUMBER - 1, index);
                EnhancedHoney currentBee = foodSources.get(index);
                EnhancedHoney neighborBee = foodSources.get(neighborIndex);
                
                searchWithStrategy(currentBee, neighborBee);
            }
            
            index = (index + 1) % FOOD_NUMBER;
        }
    }
    
    private void sendScoutBees() {
        for (EnhancedHoney honey : foodSources) {
            if (honey.getTrials() > LIMIT) {
                // Initialize new random solution
                initializeHoney(honey);
                evaluateHoney(honey);
                honey.setTrials(0);
            }
        }
    }
    
    private void evaluatePopulation() {
        for (EnhancedHoney honey : foodSources) {
            evaluateHoney(honey);
        }
    }
    
    private void evaluateHoney(EnhancedHoney honey) {
        String cacheKey = null;
        
        if (enableCaching) {
            cacheKey = Arrays.toString(honey.getNumNectar());
            if (fitnessCache.containsKey(cacheKey)) {
                honey.setFitness(fitnessCache.get(cacheKey));
                return;
            }
        }
        
        honey.computeEnhancedCost(trainSet);
        
        if (enableCaching && cacheKey != null) {
            fitnessCache.put(cacheKey, honey.getFitness());
        }
    }
    
    private void calculateProbabilities() {
        double maxFitness = foodSources.stream()
                                      .mapToDouble(EnhancedHoney::getFitness)
                                      .max().orElse(1.0);
        
        double sumFitness = foodSources.stream()
                                      .mapToDouble(EnhancedHoney::getFitness)
                                      .sum();
        
        for (EnhancedHoney honey : foodSources) {
            if (sumFitness > 0) {
                honey.setSelectionProbability(0.9 * (honey.getFitness() / maxFitness) + 0.1);
            } else {
                honey.setSelectionProbability(1.0 / FOOD_NUMBER);
            }
        }
    }
    
    private void updateGlobalBest() {
        for (EnhancedHoney honey : foodSources) {
            if (gBest == null || honey.getFitness() > gBest.getFitness()) {
                gBest = honey.deepCopy();
            }
        }
        
        // Check for improvement
        if (gBest.getFitness() > bestFitnessHistory) {
            bestFitnessHistory = gBest.getFitness();
            stagnationCount = 0;
        } else {
            stagnationCount++;
        }
    }
    
    private void updateArchive() {
        // Add non-dominated solutions to archive
        for (EnhancedHoney honey : foodSources) {
            boolean isDominated = false;
            List<EnhancedHoney> toRemove = new ArrayList<>();
            
            for (EnhancedHoney archived : archive) {
                if (archived.dominates(honey)) {
                    isDominated = true;
                    break;
                } else if (honey.dominates(archived)) {
                    toRemove.add(archived);
                }
            }
            
            if (!isDominated) {
                archive.removeAll(toRemove);
                archive.add(honey.deepCopy());
            }
        }
        
        // Maintain archive size
        if (archive.size() > config.getArchiveSize()) {
            maintainArchiveSize();
        }
    }
    
    private void maintainArchiveSize() {
        // Use crowding distance to maintain diversity
        for (EnhancedHoney solution : archive) {
            solution.calculateCrowdingDistance(archive);
        }
        
        // Sort by crowding distance and remove least diverse solutions
        archive.sort((a, b) -> Double.compare(
            a.calculateCrowdingDistance(archive), 
            b.calculateCrowdingDistance(archive)
        ));
        
        while (archive.size() > config.getArchiveSize()) {
            archive.remove(0); // Remove solution with smallest crowding distance
        }
    }
    
    private void performNSGAIISelection() {
        // Combine current population and archive
        List<EnhancedHoney> combined = new ArrayList<>(foodSources);
        combined.addAll(archive);
        
        // Perform non-dominated sorting
        fronts = performNonDominatedSorting(combined);
        
        // Select next generation
        List<EnhancedHoney> nextGeneration = new ArrayList<>();
        int frontIndex = 0;
        
        while (nextGeneration.size() < FOOD_NUMBER && frontIndex < fronts.size()) {
            List<EnhancedHoney> front = fronts.get(frontIndex);
            
            if (nextGeneration.size() + front.size() <= FOOD_NUMBER) {
                nextGeneration.addAll(front);
            } else {
                // Apply crowding distance selection for the last front
                for (EnhancedHoney solution : front) {
                    solution.calculateCrowdingDistance(front);
                }
                
                front.sort((a, b) -> Double.compare(
                    b.calculateCrowdingDistance(front), 
                    a.calculateCrowdingDistance(front)
                ));
                
                int remaining = FOOD_NUMBER - nextGeneration.size();
                nextGeneration.addAll(front.subList(0, remaining));
            }
            
            frontIndex++;
        }
        
        // Update food sources
        foodSources.clear();
        foodSources.addAll(nextGeneration);
    }
    
    private List<List<EnhancedHoney>> performNonDominatedSorting(List<EnhancedHoney> population) {
        List<List<EnhancedHoney>> fronts = new ArrayList<>();
        Map<EnhancedHoney, Integer> dominationCount = new HashMap<>();
        Map<EnhancedHoney, List<EnhancedHoney>> dominatedSolutions = new HashMap<>();
        
        // Initialize
        for (EnhancedHoney p : population) {
            dominationCount.put(p, 0);
            dominatedSolutions.put(p, new ArrayList<>());
        }
        
        // Calculate domination relationships
        for (EnhancedHoney p : population) {
            for (EnhancedHoney q : population) {
                if (p != q) {
                    if (p.dominates(q)) {
                        dominatedSolutions.get(p).add(q);
                    } else if (q.dominates(p)) {
                        dominationCount.put(p, dominationCount.get(p) + 1);
                    }
                }
            }
        }
        
        // Create fronts
        List<EnhancedHoney> currentFront = new ArrayList<>();
        for (EnhancedHoney p : population) {
            if (dominationCount.get(p) == 0) {
                currentFront.add(p);
            }
        }
        
        while (!currentFront.isEmpty()) {
            fronts.add(new ArrayList<>(currentFront));
            List<EnhancedHoney> nextFront = new ArrayList<>();
            
            for (EnhancedHoney p : currentFront) {
                for (EnhancedHoney q : dominatedSolutions.get(p)) {
                    dominationCount.put(q, dominationCount.get(q) - 1);
                    if (dominationCount.get(q) == 0) {
                        nextFront.add(q);
                    }
                }
            }
            
            currentFront = nextFront;
        }
        
        return fronts;
    }
    
    private void adaptParameters() {
        // Adaptive colony size
        double diversity = calculatePopulationDiversity();
        if (diversity < diversityThreshold) {
            adaptiveColonySize = Math.min(50, adaptiveColonySize + 2);
        } else if (diversity > diversityThreshold * 2) {
            adaptiveColonySize = Math.max(20, adaptiveColonySize - 1);
        }
        
        // Adaptive mutation rate
        if (stagnationCount > config.getEarlyStoppingPatience() / 3) {
            adaptiveMutationRate = Math.min(0.3, adaptiveMutationRate * 1.1);
        } else {
            adaptiveMutationRate = Math.max(0.05, adaptiveMutationRate * 0.99);
        }
        
        // Update configuration
        config.adaptParameters(gBest != null ? gBest.getAccuracy() : 0.0, 
                              bestFitnessHistory, diversity, stagnationCount);
    }
    
    private double calculatePopulationDiversity() {
        if (foodSources.size() < 2) return 1.0;
        
        double totalDistance = 0.0;
        int count = 0;
        
        for (int i = 0; i < foodSources.size(); i++) {
            for (int j = i + 1; j < foodSources.size(); j++) {
                totalDistance += calculateHammingDistance(
                    foodSources.get(i).getNumNectar(), 
                    foodSources.get(j).getNumNectar()
                );
                count++;
            }
        }
        
        return count > 0 ? totalDistance / count / MAX_LENGTH : 0.0;
    }
    
    private double calculateHammingDistance(int[] solution1, int[] solution2) {
        double distance = 0.0;
        for (int i = 0; i < solution1.length; i++) {
            if (solution1[i] != solution2[i]) {
                distance += 1.0;
            }
        }
        return distance;
    }
    
    private void checkConvergence() {
        if (stagnationCount > config.getEarlyStoppingPatience()) {
            System.out.println("Early stopping due to stagnation at epoch " + currentEpoch);
        }
    }
    
    private int getExclusiveRandomNumber(int max, int exclude) {
        int num;
        do {
            num = rand.nextInt(max + 1);
        } while (num == exclude);
        return num;
    }
    
    // Getters
    public EnhancedHoney getGlobalBest() { return gBest; }
    public List<EnhancedHoney> getArchive() { return new ArrayList<>(archive); }
    public int getCurrentEpoch() { return currentEpoch; }
    public double getBestFitness() { return gBest != null ? gBest.getFitness() : 0.0; }
    public List<EnhancedHoney> getFoodSources() { return new ArrayList<>(foodSources); }
}