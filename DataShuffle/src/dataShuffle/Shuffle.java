package dataShuffle;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class Shuffle {
	public static void main(String[] args) {
		String path = "E:\\ubuntu\\eclipse-workspace\\thesis\\MCRM\\data set\\Iris";
		String fileName = "iris.arff";
		for(int i = 1 ; i <= 10 ; i++) {
			shuffle(path, fileName,i, true);			
		}
	}
	
	public static void shuffle(String path,String fileName, int ittr, boolean removeMissValues) {
		FileReader reader = null;
		Scanner in = null;
		FileWriter writer = null;
		try {
			reader = new FileReader(path+"/"+fileName);
			in = new Scanner(reader);
			ArrayList<String> array = new ArrayList<>();
			ArrayList<String> header = new ArrayList<>();
			String line = null;
			while(in.hasNext()) {
				line = in.nextLine();				
				if(line.trim().equals("") || line.startsWith("%") || line.startsWith("@")) {
					header.add(line);
				}else if(line.contains("?") && removeMissValues) {
					continue;
				}else {
					array.add(line);					
				}
			}
			File f = new File(path+"/10");
			if(!f.exists()) {
				f.mkdir();
			}
			Collections.shuffle(array);			
			writer = new FileWriter(path+"/10/"+ittr+".arff");
			for(String str : header) {
				writer.write(str+"\n");
			}
			for(String str : array) {
				writer.write(str+"\n");
			}
		} catch (FileNotFoundException e) {			
			e.printStackTrace();
		} catch (IOException e) {			
			e.printStackTrace();
		}finally {			
			try {
				writer.close();
				reader.close();
			} catch (IOException e) {				
				e.printStackTrace();
			}
			in.close();
		}
	}
}
