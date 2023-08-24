/**
 * ClassName:YoloDetector.java
 * Date:2022年4月8日
 */
package com.idata.vision;

//import static org.bytedeco.javacpp.opencv_imgproc.putText;
//import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

import javax.imageio.ImageIO;

import org.bytedeco.javacpp.avcodec;
import org.bytedeco.javacpp.avutil;
//import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter.ToMat;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import net.coobird.thumbnailator.Thumbnails;

/**
 * Creater:SHAO Gaige
 * Description:yolo算法识别类
 * Log:
 */
public class YoloDetector {
	
	//配置文件对象
	private YoloConfig YOLO_CONFIG;
	/*parent Dataset folder "DATA_DIR" contains two subfolder "images" and "annotations" */
	private String MODEL_DIR = "";
	//神经网络模型对象
	private ComputationGraph NETWORK;
	//停止标识
	private boolean stop = false;
	
	public YoloDetector(YoloConfig yolo_config,String model_dir)
	{
		this.YOLO_CONFIG = yolo_config;
		this.MODEL_DIR = model_dir;
        File net = new File(model_dir);
        boolean modelexists = net.exists() && !net.isDirectory();
        if (modelexists) 
        {
            try 
            {
                NETWORK = ModelSerializer.restoreComputationGraph(model_dir);
                //System.out.println(NETWORK.summary());
                System.out.println(model_dir+" yolo模型加载成功...");
                
                //setModelClasses(classes.split("\\,"));
            }
            catch (IOException ex)
            {
                System.out.println(ex.getMessage());
            }
        }
        else 
        {
        	System.out.println("Can't find model file "+model_dir+"\n  "
                    + "Please Train the dataset first to provide the model file");
        }
	}
	
	public void detectVideo(String inputFile,String outputFile,double detectionthreshold)
	{
		try
		{
			Frame[] videoFrame = new Frame[1];
			Mat[] v = new Mat[1];
			
			File f = new File(inputFile);
			FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(f);
			grabber.start();
			
			double frameRate = grabber.getFrameRate();
			System.out.println("The inputted video clip has " + grabber.getLengthInFrames() + " frames");
			 
			System.out.println("Frame rate " + frameRate + "fps");
			
			FFmpegFrameRecorder recorder = new FFmpegFrameRecorder(outputFile, grabber.getImageWidth(),grabber.getImageHeight(), 0);
			recorder.setVideoCodec(avcodec.AV_CODEC_ID_H264);
			//recorder.setAudioChannels(1);
			//recorder.setInterleaved(true);
			recorder.setFormat("mp4");
			recorder.setFrameRate(grabber.getFrameRate());
			recorder.setPixelFormat(avutil.AV_PIX_FMT_YUV420P); // yuv420p
			int bitrate = grabber.getVideoBitrate();
			if (bitrate == 0) {
				bitrate = grabber.getAudioBitrate();
			}
			recorder.setVideoBitrate(bitrate);
			
			recorder.start();
			
			while (!stop)
			{
				videoFrame[0] = grabber.grab();
				if (videoFrame[0] == null) {
					stop = true;
					break;
				}
				v[0] = new OpenCVFrameConverter.ToMat().convert(videoFrame[0]);
				if (v[0] == null) {
					continue;
				}

				//图像大小的特殊处理
				Mat simage = v[0];
	        	if(v[0].arrayHeight()!=416 || v[0].arrayWidth()!=416)
	        	{
	        		System.out.println(v[0].arrayHeight()+"x"+v[0].arrayWidth()+"重新调整大小后进行识别");
	        		BufferedImage  _simage = matToBufferedImage(v[0]);
	        		_simage = Thumbnails.of(_simage).width(416).height(416).asBufferedImage();
	        		BufferedImage outputImage = new BufferedImage(416, 416, BufferedImage.TYPE_INT_RGB);
	            	Graphics g = outputImage.getGraphics();
	            	g.setColor(Color.white);
	            	g.fillRect(0, 0, 416, 416);
	                g.drawImage(_simage, 0, 0, null);
	                g.dispose();
	                File _f = new File(outputFile);
	                String temp = _f.getParent()+"t_"+_f.getName()+".png";
	                _f = new File(temp);
	        		ImageIO.write(outputImage, "png", f);
	        		simage = opencv_imgcodecs.imread(temp);
	                //Mat _simage = bufferedImageToMat(outputImage);
	        		//删除
	        		f.delete();
	        	}
				//==============
				detectImage(simage,v[0],detectionthreshold);
				
				OpenCVFrameConverter.ToIplImage cvCoreMat = new OpenCVFrameConverter.ToIplImage();
				Frame frame = cvCoreMat.convert(v[0]);
				recorder.record(frame);
			}
			recorder.close();
			recorder.release();
			grabber.close();
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
		
	}
	
	public String detectJson(String imageFile,double detectionthreshold)
	{
		try
		{
			Mat oimage = opencv_imgcodecs.imread(imageFile);
			Mat simage = oimage;
			if(oimage.arrayHeight()!=416 || oimage.arrayWidth()!=416)
	    	{
				System.out.println(oimage.arrayHeight()+"x"+oimage.arrayWidth()+"重新调整大小后进行识别");
	    		BufferedImage  _simage = matToBufferedImage(oimage);
	    		_simage = Thumbnails.of(_simage).width(416).height(416).asBufferedImage();
	    		BufferedImage outputImage = new BufferedImage(416, 416, BufferedImage.TYPE_INT_RGB);
	        	Graphics g = outputImage.getGraphics();
	        	g.setColor(Color.white);
	        	g.fillRect(0, 0, 416, 416);
	            g.drawImage(_simage, 0, 0, null);
	            g.dispose();
	            File f = new File(imageFile);
	            String temp = f.getParent()+"t_"+f.getName();
	            f = new File(temp);
	    		ImageIO.write(outputImage, "png", f);
	    		simage = opencv_imgcodecs.imread(temp);
	            //Mat _simage = bufferedImageToMat(outputImage);
	    		//删除
	    		f.delete();
	    	}
			return detectJson(simage,detectionthreshold);
		}
		catch(Exception e)
		{
			e.printStackTrace();
			return "";
		}
	}
	
	private String detectJson(Mat image, double detectionthreshold)
	{
		long start = System.currentTimeMillis();
		List<DetectedObject> objects = detectObject(image,detectionthreshold);
		long end = System.currentTimeMillis();
		JsonArray rs = new JsonArray();
		for (DetectedObject obj : objects)
        {
			JsonObject r = new JsonObject();
			
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            //System.out.println(obj.getConfidence());
            int predictedClass = obj.getPredictedClass();
            //System.out.println("Predicted class " + YOLO_CONFIG.getCLASSES()[predictedClass]);  
            int x1 = (int) Math.round(YOLO_CONFIG.INPUT_WIDTH * xy1[0] / YOLO_CONFIG.GRID_WIDTH);
            x1 = x1<0?2:x1;
            int y1 = (int) Math.round(YOLO_CONFIG.INPUT_HEIGHT * xy1[1] / YOLO_CONFIG.GRID_HEIGHT);
            y1 = y1<0?2:y1;
            int x2 = (int) Math.round(YOLO_CONFIG.INPUT_WIDTH * xy2[0] / YOLO_CONFIG.GRID_WIDTH);
            x2 = x2>YOLO_CONFIG.INPUT_WIDTH-2?YOLO_CONFIG.INPUT_WIDTH-2:x2;
            int y2 = (int) Math.round(YOLO_CONFIG.INPUT_HEIGHT * xy2[1] / YOLO_CONFIG.GRID_HEIGHT);
            y2 = y2>YOLO_CONFIG.INPUT_HEIGHT-2?YOLO_CONFIG.INPUT_HEIGHT:y2;
            //处理不同尺寸的图片
            double dx = image.arrayWidth()*1.00/YOLO_CONFIG.INPUT_WIDTH;
            double dy = image.arrayHeight()*1.00/YOLO_CONFIG.INPUT_HEIGHT;
            if(image.arrayWidth()>image.arrayHeight())
            {
            	 dx = image.arrayWidth()*1.00/YOLO_CONFIG.INPUT_WIDTH;
                 dy = dx;
            }
            else
            {
            	 dy = image.arrayHeight()*1.00/YOLO_CONFIG.INPUT_HEIGHT;
            	 dx = dy;
            }
            x1 = (int)Math.floor(x1*dx);
            y1 = (int)Math.floor(y1*dy);
            x2 = (int)Math.ceil(x2*dx);
            y2 = (int)Math.ceil(y2*dy);
            x2 = x2>image.arrayWidth()-2?image.arrayWidth()-2:x2;
            y2 = y2>image.arrayHeight()-2?image.arrayHeight()-2:y2;

            r.addProperty("classname", YOLO_CONFIG.getCLASSESNAME()[predictedClass]);
            r.addProperty("precision", obj.getConfidence());
            r.addProperty("time", end-start);
            r.addProperty("mark", "onedata.vision.shaogaige");
            r.addProperty("x1", x1);
            r.addProperty("y1", y1);
            r.addProperty("x2", x2);
            r.addProperty("y2", y2);
            
            rs.add(r);
        }
		return rs.toString();
	}
	
	public double getDetectPrecision(String imagePath)
	{
		File file = new File(imagePath);
		if(!file.isDirectory())
		{
			return 0.00;
		}
		else
		{
			int cout = 0;
			int decout = 0;
			File[] files = file.listFiles();
			for (File f : files)
			{
				if(f.isFile())
				{
					cout++;
					Mat image = opencv_imgcodecs.imread(f.getAbsolutePath());
					List<DetectedObject> obs = detectObject(image,0.29);
					if(obs.size()<1)
		        	{
		        		System.out.println(image.arrayHeight());
		            	System.out.println(image.arrayWidth());
		            	if(image.arrayHeight()!=416 || image.arrayWidth()!=416)
		            	{
		            		BufferedImage  simage = matToBufferedImage(image);
		            		try {
								simage = Thumbnails.of(simage).width(416).height(416).asBufferedImage();
							} catch (IOException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							}
		            		Mat _simage = bufferedImageToMat(simage);
		            		obs = detectObject(_simage,0.29);
		            	}
		        	}
					if(obs.size()>0)
					{
						decout++;
					}
				}
			}
			return decout*1.00/cout;
		}
	}
	
	
	private List<DetectedObject> detectObject(Mat image, double detectionthreshold)
	{
		Yolo2OutputLayer yout = (Yolo2OutputLayer) NETWORK.getOutputLayer(0);
        NativeImageLoader loader = new NativeImageLoader(YOLO_CONFIG.INPUT_WIDTH, YOLO_CONFIG.INPUT_HEIGHT, YOLO_CONFIG.CHANNELS);//, new ColorConversionTransform(COLOR_BGR2RGB)
        INDArray ds = null;
        try 
        {
            ds = loader.asMatrix(image);
        } 
        catch (IOException ex)
        {
        	System.out.println(ex.getMessage());
        }
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(ds);
        INDArray results = NETWORK.outputSingle(ds);
        List<DetectedObject> objs = yout.getPredictedObjects(results, detectionthreshold);
        List<DetectedObject> objects = NonMaxSuppression.getObjects(objs);
        return objects;
	}

    private Mat detectImage(Mat simage,Mat image, double detectionthreshold)
    {
        List<DetectedObject> objects = detectObject(simage,detectionthreshold);
        BufferedImage bufImg = matToBufferedImage(image);
        bufImg = drawBoxes(bufImg, objects,false);//use objs to see the use of the NonMax Suppression algorithm
        image = bufferedImageToMat(bufImg);
        return image;
        //opencv_imgcodecs.imwrite("D:/onedata/out.png",image);
    }
    
    public void detectImage(String imageFile,String outputFile,double detectionthreshold)
    {
        try
        {
        	Mat oimage = opencv_imgcodecs.imread(imageFile);
        	Mat simage = oimage;
        	if(oimage.arrayHeight()!=416 || oimage.arrayWidth()!=416)
        	{
        		System.out.println(oimage.arrayHeight()+"x"+oimage.arrayWidth()+"重新调整大小后进行识别");
        		BufferedImage  _simage = matToBufferedImage(oimage);
        		_simage = Thumbnails.of(_simage).width(416).height(416).asBufferedImage();
        		BufferedImage outputImage = new BufferedImage(416, 416, BufferedImage.TYPE_INT_RGB);
            	Graphics g = outputImage.getGraphics();
            	g.setColor(Color.white);
            	g.fillRect(0, 0, 416, 416);
                g.drawImage(_simage, 0, 0, null);
                g.dispose();
                File f = new File(outputFile);
                String temp = f.getParent()+"t_"+f.getName();
                f = new File(temp);
        		ImageIO.write(outputImage, "png", f);
        		simage = opencv_imgcodecs.imread(temp);
                //Mat _simage = bufferedImageToMat(outputImage);
        		//删除
        		f.delete();
        	}
        	List<DetectedObject> objects = detectObject(simage,detectionthreshold);
            BufferedImage bufImg = matToBufferedImage(oimage);
            bufImg = drawBoxes(bufImg, objects,false);//use objs to see the use of the NonMax Suppression algorithm
        	//opencv_imgcodecs.imwrite(outputFile,oimage);
        	File outputfile = new File(outputFile);
			ImageIO.write(bufImg,outputFile.substring(outputFile.lastIndexOf(".")+1),outputfile);
		}
        catch (Exception e)
        {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println(outputFile+" 文件写出失败！！！");
		}
    }
    
    public void detectImage(String imageFile,String outputFile,double detectionthreshold,boolean showConfidence,boolean showTime,String mark)
    {
        try
        {
        	Mat oimage = opencv_imgcodecs.imread(imageFile);
        	Mat simage = oimage;
        	if(oimage.arrayHeight()!=416 || oimage.arrayWidth()!=416)
        	{
        		System.out.println(oimage.arrayHeight()+"x"+oimage.arrayWidth()+"重新调整大小后进行识别");
        		BufferedImage  _simage = matToBufferedImage(oimage);
        		_simage = Thumbnails.of(_simage).width(416).height(416).asBufferedImage();
        		BufferedImage outputImage = new BufferedImage(416, 416, BufferedImage.TYPE_INT_RGB);
            	Graphics g = outputImage.getGraphics();
            	g.setColor(Color.white);
            	g.fillRect(0, 0, 416, 416);
                g.drawImage(_simage, 0, 0, null);
                g.dispose();
                File f = new File(outputFile);
                String temp = f.getParent()+"t_"+f.getName();
                f = new File(temp);
        		ImageIO.write(outputImage, "png", f);
        		simage = opencv_imgcodecs.imread(temp);
                //Mat _simage = bufferedImageToMat(outputImage);
        		//删除
        		f.delete();
        	}
        	long start = System.currentTimeMillis();
        	List<DetectedObject> objects = detectObject(simage,detectionthreshold);
        	long end = System.currentTimeMillis();
            BufferedImage bufImg = matToBufferedImage(oimage);
            bufImg = drawBoxes(bufImg, objects,showConfidence);//use objs to see the use of the NonMax Suppression algorithm
            putAttach(bufImg,showTime,end-start,mark);
        	File outputfile = new File(outputFile);
        	File dir = new File(outputfile.getParent());
    		if (!dir.exists()) {
    			dir.mkdirs();
    		}
			ImageIO.write(bufImg,outputFile.substring(outputFile.lastIndexOf(".")+1),outputfile);
		}
        catch (Exception e)
        {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println(outputFile+" 文件写出失败！！！");
		}
    }
    
    private BufferedImage drawBoxes(BufferedImage bufimage, List<DetectedObject> objects,boolean showConfidence)
	{
        for (DetectedObject obj : objects)
        {
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            //System.out.println(obj.getConfidence());
            System.out.println(bufimage.getWidth()+" x "+bufimage.getHeight());
            int predictedClass = obj.getPredictedClass();
            //System.out.println("Predicted class " + YOLO_CONFIG.getCLASSES()[predictedClass]);
            int x1 = (int) Math.round(YOLO_CONFIG.INPUT_WIDTH * xy1[0] / YOLO_CONFIG.GRID_WIDTH);
            x1 = x1<0?2:x1;
            int y1 = (int) Math.round(YOLO_CONFIG.INPUT_HEIGHT * xy1[1] / YOLO_CONFIG.GRID_HEIGHT);
            y1 = y1<0?2:y1;
            int x2 = (int) Math.round(YOLO_CONFIG.INPUT_WIDTH * xy2[0] / YOLO_CONFIG.GRID_WIDTH);
            x2 = x2>YOLO_CONFIG.INPUT_WIDTH-2?YOLO_CONFIG.INPUT_WIDTH-2:x2;
            int y2 = (int) Math.round(YOLO_CONFIG.INPUT_HEIGHT * xy2[1] / YOLO_CONFIG.GRID_HEIGHT);
            y2 = y2>YOLO_CONFIG.INPUT_HEIGHT-2?YOLO_CONFIG.INPUT_HEIGHT:y2;
            //处理不同尺寸的图片
            //System.out.println(x1+","+y1+","+x2+","+y2);
            double dx = bufimage.getWidth()*1.00/YOLO_CONFIG.INPUT_WIDTH;
            double dy = bufimage.getHeight()*1.00/YOLO_CONFIG.INPUT_HEIGHT;
            if(bufimage.getWidth()>bufimage.getHeight())
            {
            	 dx = bufimage.getWidth()*1.00/YOLO_CONFIG.INPUT_WIDTH;
                 dy = dx;
            }
            else
            {
            	 dy = bufimage.getHeight()*1.00/YOLO_CONFIG.INPUT_HEIGHT;
            	 dx = dy;
            }
            //double dx = bufimage.getWidth()*1.00/YOLO_CONFIG.INPUT_WIDTH;
            //double dy = bufimage.getHeight()*1.00/YOLO_CONFIG.INPUT_HEIGHT;
            x1 = (int)Math.floor(x1*dx);
            y1 = (int)Math.floor(y1*dy);
            x2 = (int)Math.ceil(x2*dx);
            y2 = (int)Math.ceil(y2*dy);
            x2 = x2>bufimage.getWidth()-2?bufimage.getWidth()-2:x2;
            y2 = y2>bufimage.getHeight()-2?bufimage.getHeight()-2:y2;
            //System.out.println(x1+","+y1+","+x2+","+y2);
            //rectangle(image, new opencv_core.Point(x1, y1), new opencv_core.Point(x2, y2), opencv_core.Scalar.RED);
            //putText(image, YOLO_CONFIG.getCLASSES()[predictedClass], new opencv_core.Point(x1 + 2, y2 - 2), 1, .8, opencv_core.Scalar.RED);
            bufimage = putClass(bufimage,YOLO_CONFIG.getCLASSESNAME()[predictedClass],Math.abs(x1),x2,y1,y2,showConfidence,obj.getConfidence());
        }
        return bufimage;
    }
    
    private BufferedImage putClass(BufferedImage bufImg,String text,int x1,int x2,int y1,int y2,boolean showConfidence,double confidence)
    {
		Graphics2D g = bufImg.createGraphics();
        g.drawImage(bufImg, 0, 0, bufImg.getWidth(),bufImg.getHeight(), null);
        g.setColor(Color.RED);
        //文字边缘平滑 by zhengkai.blog.csdn.net
        //g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING,RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        //图片边缘平滑 by zhengkai.blog.csdn.net
        //g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        //绘制矩形框
        g.drawRect(x1, y1,Math.abs(x2-x1) ,Math.abs(y2-y1));
        //设置字体
        Font font = new Font("微软雅黑", Font.PLAIN, 12); 
        g.setFont(font);
        //设置水印的坐标
        g.drawString(text, x1+2, y2-2);
        String context = "";
        if(showConfidence)
        {
        	context = "置信度:"+String.format("%.1f",confidence*100)+"%";
        }
        font = new Font("微软雅黑", Font.PLAIN, 10); 
        g.setFont(font);
        g.setColor(Color.GREEN);
        g.drawString(context,x1+2, y1+12);
        g.dispose();
        return bufImg;
    }
    
    private BufferedImage putAttach(BufferedImage bufImg,boolean showTime,long time,String mark)
    {
		Graphics2D g = bufImg.createGraphics();
        g.drawImage(bufImg, 0, 0, bufImg.getWidth(),bufImg.getHeight(), null);
        g.setColor(Color.BLUE);
        //设置字体
        Font font = new Font("微软雅黑", Font.PLAIN, 12); 
        g.setFont(font);
        //设置水印的坐标
        String text = "";
        
        if(showTime)
        {
        	text +="耗时："+time+" ms  ";
        }
        text += mark;
        g.drawString(text,10, bufImg.getHeight()-4);
        g.dispose();
        return bufImg;
    }
    
    
    public BufferedImage matToBufferedImage(org.bytedeco.javacpp.opencv_core.Mat image)
    {
    	ToMat convert= new ToMat();
     	Frame frame= convert.convert(image);
     	Java2DFrameConverter java2dFrameConverter = new Java2DFrameConverter();
     	BufferedImage bufferedImage = java2dFrameConverter.convert(frame);
     	return bufferedImage;
    }
    
    public org.bytedeco.javacpp.opencv_core.Mat bufferedImageToMat(BufferedImage bi) {
        OpenCVFrameConverter.ToMat cv = new OpenCVFrameConverter.ToMat();
        return cv.convertToMat(new Java2DFrameConverter().convert(bi)); 
    }

	public YoloConfig getYOLO_CONFIG() {
		return YOLO_CONFIG;
	}

	public String getMODEL_DIR() {
		return MODEL_DIR;
	}

	public ComputationGraph getNETWORK() {
		return NETWORK;
	}

	public boolean isStop() {
		return stop;
	}

	public void setStop(boolean stop) {
		this.stop = stop;
	}

}
