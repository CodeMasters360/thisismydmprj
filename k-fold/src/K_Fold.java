import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.LinkedList;
import java.util.Scanner;

public class K_Fold {

	public static void main(String[] args) throws IOException {
		String dataSetMainDirectory = "Iris";
		String path = String.format("E:\\ubuntu\\eclipse-workspace\\thesis\\MCRM\\data set\\%s",dataSetMainDirectory);
		new File(path+"/"+dataSetMainDirectory).mkdir();
		for(int i = 0 ; i < 10 ; i++) {
			new File(path+"/"+dataSetMainDirectory+"/shuffle-"+(i+1)).mkdir();
			String sourcePath = String.format("%s/10/%d.arff", path,(i+1));
			String distPath = String.format("%s/%s/shuffle-%d", path,dataSetMainDirectory,(i+1));
			discreateData(sourcePath, distPath, 10);
		}
	}
	private static void discreateData(String sourcePath, String distPath, int k) throws IOException {
		LinkedList[] arr = new LinkedList[k];
		LinkedList<String> header = new LinkedList<>();
		LinkedList<String> data = new LinkedList<>();		
		FileReader reader = new FileReader(sourcePath);
		Scanner input = new Scanner(reader);
		while(input.hasNext()) {
			String line = input.nextLine();
			if(line.trim().equals("") || line.startsWith("%") || line.startsWith("@")) {
				header.add(line);
			}else {
				data.add(line);				
			}
		}
		int foldSize = data.size() / k;
		int counter = -1;
		int remain = data.size() % k;
		for(int i = 0 ; i < k ; i++) {
			arr[i] = new LinkedList<String>();
			for(int j = 0 ; j < foldSize ; j++) {
				counter++;
				arr[i].add(data.get(counter));
			}
			if(remain > 0) {
				counter++;
				arr[i].add(data.get(counter));
				remain--;
			}
		}
		reader.close();
		input.close();
		for(int i = 0 ; i < k ; i++) {
			File f = new File(String.format("%s/fold-%d",distPath,i+1));
			if(!f.exists()) {
				f.mkdir();
			}
			FileWriter trainWriter = new FileWriter(String.format("%s/fold-%d/%s",distPath,i+1,"train.arff"));
			for(String str : header) {
				trainWriter.write(str+"\n");
			}
			for(int j = 0 ; j < k ; j++) {
				if(j != i) {
					for(String str : (LinkedList<String>)arr[j]) {
						trainWriter.write(str+"\n");
					}
				}
			}
			trainWriter.close();			
			FileWriter testWriter = new FileWriter(String.format("%s/fold-%d/%s",distPath,i+1,"test.arff"));
			for(String str : header) {
				testWriter.write(str+"\n");
			}
			for(String str : (LinkedList<String>)arr[i]) {
				testWriter.write(str+"\n");
			}
			testWriter.close();
		}		
	}
}
