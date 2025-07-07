package utils;

/**
 * Enhanced Data class with additional capabilities
 */
public class DataEnhanced implements Comparable<DataEnhanced> {
    public static int sortFeature = 0;
    private static int numOfNumericalFeatures = 0;
    private static int labelInterval = 0;
    private static double[] numericalFeatureMaxs;
    private static double[] numericalFeatureMins;
    
    public double[] numericalFeatures;
    public int label;
    
    public DataEnhanced(double[] numericalFeatures, int label) {
        this.numericalFeatures = numericalFeatures;
        this.label = label;
    }
    
    public DataEnhanced() {
        this.numericalFeatures = new double[numOfNumericalFeatures];
        this.label = 0;
    }
    
    @Override
    public int compareTo(DataEnhanced other) {
        if (this.numericalFeatures[sortFeature] < other.numericalFeatures[sortFeature]) {
            return -1;
        } else if (this.numericalFeatures[sortFeature] > other.numericalFeatures[sortFeature]) {
            return 1;
        }
        return 0;
    }
    
    // Static methods for configuration
    public static void setNumOfNumericalFeatures(int numOfNumericalFeatures) {
        DataEnhanced.numOfNumericalFeatures = numOfNumericalFeatures;
        numericalFeatureMaxs = new double[numOfNumericalFeatures];
        numericalFeatureMins = new double[numOfNumericalFeatures];
    }
    
    public static void setNumericalFeatureMaxs(double[] numericalFeatureMaxs) {
        DataEnhanced.numericalFeatureMaxs = numericalFeatureMaxs;
    }
    
    public static void setNumericalFeatureMins(double[] numericalFeatureMins) {
        DataEnhanced.numericalFeatureMins = numericalFeatureMins;
    }
    
    public static void setNumericalFeatureMax(int f, double max) {
        DataEnhanced.numericalFeatureMaxs[f] = max;
    }
    
    public static void setNumericalFeatureMin(int f, double min) {
        DataEnhanced.numericalFeatureMins[f] = min;
    }
    
    public static void setLabelInterval(int intv) {
        labelInterval = intv;
    }
    
    public static int getNumOfNumericalFeatures() {
        return numOfNumericalFeatures;
    }
    
    public static int getLabelInterval() {
        return labelInterval;
    }
    
    public static double getNumericalFeatureMaxs(int n) {
        return numericalFeatureMaxs[n];
    }
    
    public static double getNumericalFeatureMins(int n) {
        return numericalFeatureMins[n];
    }
    
    public static double[] getNumericalFeatureMaxs() {
        return numericalFeatureMaxs;
    }
    
    public static double[] getNumericalFeatureMins() {
        return numericalFeatureMins;
    }
    
    /**
     * Calculate Euclidean distance between two data points
     */
    public double distance(DataEnhanced other) {
        double sum = 0.0;
        for (int i = 0; i < numericalFeatures.length; i++) {
            double diff = this.numericalFeatures[i] - other.numericalFeatures[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
    
    /**
     * Normalize features using min-max normalization
     */
    public void normalize() {
        for (int i = 0; i < numericalFeatures.length; i++) {
            double range = numericalFeatureMaxs[i] - numericalFeatureMins[i];
            if (range > 0) {
                numericalFeatures[i] = (numericalFeatures[i] - numericalFeatureMins[i]) / range;
            }
        }
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("DataEnhanced{features=[");
        for (int i = 0; i < numericalFeatures.length; i++) {
            sb.append(String.format("%.3f", numericalFeatures[i]));
            if (i < numericalFeatures.length - 1) sb.append(", ");
        }
        sb.append("], label=").append(label).append("}");
        return sb.toString();
    }
}