/**
 * ClassName:YOLOTEST.java
 * Date:2022年4月11日
 */
package com.idata.test;

import java.io.File;
import java.util.List;

import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;

import com.idata.vision.YoloConfig;
import com.idata.vision.YoloDetector;
import com.idata.vision.YoloTrainer;

/**
 * Creater:SHAO Gaige
 * Description:
 * Log:
 */
public class YOLOTEST {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		String modelpath = "D:\\onedata\\in\\M300_model.data";
		String imagepath = "D:\\data\\deeplearning\\M300\\images-original\\uav15.jpg";
		//imagepath = "D:\\onedata\\in\\_1_uav404.png";
		String outputFile = "D:\\onedata\\in\\_uav15.png";
		
		//System.out.println(imagepath.substring(imagepath.lastIndexOf(".")+1));
		
		YoloConfig config = new YoloConfig();
//		
		YoloDetector yolo = new YoloDetector(config,modelpath);
		//yolo.detectImage(imagepath, outputFile, 0.2);
//		
		yolo.detectImage(imagepath, outputFile, 0.1,true,true,"邵改革");
//		Mat oimage = opencv_imgcodecs.imread(imagepath);
//		String s = yolo.detectJson(oimage,0.001);
//		System.out.println(s);
		
		//String inputFile = "D:\\onedata\\in\\in.mp4";
		//outputFile = "D:\\onedata\\in\\_in.mp4";
		//yolo.detectVideo(inputFile, outputFile, 0.4);
		
//		String imagePath = "D:\\data\\deeplearning\\M300\\images-original\\";
//		double d = yolo.getDetectPrecision(imagePath);
//		System.out.println(d);
		
//		String inpath = "D:\\data\\deeplearning\\M300\\images-original\\";
//		String outpath = "D:\\onedata\\out\\M300_detect2\\";
//		File file = new File(inpath);
//		if(!file.isDirectory())
//		{
//			return;
//		}
//		else
//		{
//			File[] files = file.listFiles();
//			for (File f : files)
//			{
//				if(f.isFile())
//				{
//					String name = f.getName();
//					yolo.detectImage(f.getAbsolutePath(), outpath+name, 0.35,true,true,"云商视图");
//					
//				}
//			}
//		}
		
		
		//train();

	}
	
	
	
	public static void train() throws Exception
	{
		YoloTrainer  yt = new YoloTrainer();
		YoloConfig config = new YoloConfig();
		String modelpath = "D:\\onedata\\in\\M300_model.data";
		String DATA_DIR = "D:\\data\\deeplearning\\M300";
		yt.train(config, DATA_DIR);
		yt.savemodel(modelpath);
	}

}
