/**
 * ClassName:YoloTrainer.java
 * Date:2022年4月8日
 */
package com.idata.vision;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Date;
import java.util.Random;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;

/**
 * Creater:SHAO Gaige
 * Description:yolo训练模型类
 * Log:
 */
public class YoloTrainer implements Runnable{
	
	//配置文件对象
	private YoloConfig YOLO_CONFIG;
	/*parent Dataset folder "DATA_DIR" contains two subfolder "images" and "annotations" */
	private String DATA_DIR = "";
	//UIServer对象
	private UIServer uiServer;
	//停止标识
	private boolean flag = true;
	//uuid
	private String uuid = "yolotrainer";
	//model
	private ComputationGraph model;
	//index
	private int index = 0;
	//模型存储路径
	private String modelpath = "";
	
	public YoloTrainer()
	{
		this.uuid = this.uuid+ new Date().getTime();
	}
	
	public YoloTrainer(UIServer uiServer,String uuid)
	{
		this.uiServer = uiServer;
		this.uuid = uuid;
	}
	
	public YoloTrainer(YoloConfig yoloconfig, String data_dir,String modelpath)
	{
		this.uuid = this.uuid+ new Date().getTime();
		this.setYOLO_CONFIG(yoloconfig);
		this.DATA_DIR = data_dir;
		this.modelpath = modelpath;
	}
	
	
	public boolean train(YoloConfig yoloconfig, String data_dir)
	{
		this.setYOLO_CONFIG(yoloconfig);
		if(YOLO_CONFIG == null)
		{
			System.out.println("yoloconfig 对象为空！！！");
			return false;
		}
		if(data_dir == null || "".equalsIgnoreCase(data_dir))
		{
			System.out.println("data_dir 数据集路径为空！！！");
			return false;
		}
		System.out.println("数据集路径："+data_dir);
		yoloconfig.setDATA_DIR(data_dir);
		this.setDATA_DIR(data_dir);
		
		/* ###########################################    */
		
		try
		{
			Random rng = new Random(YOLO_CONFIG.SEED);

	        //Initialize the user interface backend, it is just as tensorboard.
	        //it starts at http://localhost:9000
	        if(this.uiServer == null)
	        {
	        	this.uiServer = UIServer.getInstance();
	        }

	        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
	        StatsStorage statsStorage = new InMemoryStatsStorage();

	        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
	        uiServer.attach(statsStorage);

	        File imageDir = new File(DATA_DIR, "images");

	        System.out.println("Load data...");
	        RandomPathFilter pathFilter = new RandomPathFilter(rng) {
	            @Override
	            protected boolean accept(String name) {
	                name = name.replace("/images/", "/annotations/").replace(".jpg", ".xml");
	                //System.out.println("Name " + name);
	                try {
	                    return new File(new URI(name)).exists();
	                } catch (URISyntaxException ex) {
	                    throw new RuntimeException(ex);
	                }
	            }
	        };

	        InputSplit[] data = new FileSplit(imageDir, NativeImageLoader.ALLOWED_FORMATS, rng).sample(pathFilter, 0.9, 0.1);
	        InputSplit trainData = data[0];
	        InputSplit testData = data[1];

	        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(YOLO_CONFIG.INPUT_HEIGHT, YOLO_CONFIG.INPUT_WIDTH, 
	        		YOLO_CONFIG.CHANNELS,YOLO_CONFIG.GRID_HEIGHT, YOLO_CONFIG.GRID_WIDTH, new VocLabelProvider(DATA_DIR));
	        recordReaderTrain.initialize(trainData);

	        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(YOLO_CONFIG.INPUT_HEIGHT, YOLO_CONFIG.INPUT_WIDTH, 
	        		YOLO_CONFIG.CHANNELS,YOLO_CONFIG.GRID_HEIGHT, YOLO_CONFIG.GRID_WIDTH, new VocLabelProvider(DATA_DIR));
	        recordReaderTest.initialize(testData);

	        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, YOLO_CONFIG.BATCH_SIZE, 1, 1, true);
	        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));

	        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
	        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));

	        ComputationGraph pretrained = (ComputationGraph) TinyYOLO.builder().build().initPretrained();

	        INDArray priors = Nd4j.create(YOLO_CONFIG.PRIOR_BOXES);
	        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
	                .seed(YOLO_CONFIG.SEED)
	                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
	                .gradientNormalizationThreshold(1.0)
	                .updater(new RmsProp(YOLO_CONFIG.LEARNIGN_RATE))
	                .activation(Activation.IDENTITY).miniBatch(true)
	                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
	                .build();

	        model = new TransferLearning.GraphBuilder(pretrained)
	                .fineTuneConfiguration(fineTuneConf)
	                .setInputTypes(InputType.convolutional(YOLO_CONFIG.INPUT_HEIGHT, YOLO_CONFIG.INPUT_WIDTH, YOLO_CONFIG.CHANNELS))
	                .removeVertexKeepConnections("conv2d_9")
	                .addLayer("convolution2d_9",
	                        new ConvolutionLayer.Builder(1, 1)
	                                .nIn(1024)
	                                .nOut(YOLO_CONFIG.BOXES_NUMBER * (5 + YOLO_CONFIG.CLASSES_NUMBER))
	                                .stride(1, 1)
	                                .convolutionMode(ConvolutionMode.Same)
	                                .weightInit(WeightInit.UNIFORM)
	                                .hasBias(false)
	                                .activation(Activation.IDENTITY)
	                                .build(), "leaky_re_lu_8")
	                .addLayer("outputs",
	                        new Yolo2OutputLayer.Builder()
	                                .lambbaNoObj(YOLO_CONFIG.LAMDBA_NO_OBJECT)
	                                .lambdaCoord(YOLO_CONFIG.LAMDBA_COORD)
	                                .boundingBoxPriors(priors)
	                                .build(), "convolution2d_9")
	                .setOutputs("outputs")
	                .build();

	        System.out.println("\n Model Summary \n" + model.summary());

	        System.out.println("Train model...");
	        //model.setListeners(new ScoreIterationListener(1));//print score after each iteration on stout 
	        model.setListeners(new StatsListener(statsStorage));// visit http://localhost:9000 to track the training process
	        for (int i = 0; i < YOLO_CONFIG.EPOCHS; i++) 
	        {
	            train.reset();
	            while (train.hasNext() && this.flag) 
	            {
	                model.fit(train.next());
	            }
	            index = i+1;
	            System.out.println("*** Completed epoch "+i+" ***"+new Date().toString());
	            if(!this.flag)
	            {
	            	System.out.println("*** "+this.uuid+" 被强行中断执行 ！！！***");
	            	this.uiServer.stop();
	            	return false;
	            }
	        }

	        System.out.println("*** Training Done ***");
	        /* ###########################################    */
			return true;
		}
		catch(Exception e)
		{
			e.printStackTrace();
			return false;
		}
	}
	
	public void breaktrain()
	{
		this.flag = false;
	}
	
	public void savemodel(String model_outpath)
	{
        try 
        {
        	System.out.println("*** Saving Model ***");
    		File f = new File(model_outpath);
    		if(f.isDirectory())
    		{
    			model_outpath += "model_"+new Date().getTime()+".data";
    			f = new File(model_outpath);
    		}
    		if(!f.getParentFile().exists())
    		{
    			f.mkdirs();
    		}
            //输出路径文件
			ModelSerializer.writeModel(this.model, model_outpath, false);
			this.YOLO_CONFIG.setMODEL_PATH(model_outpath);
	        this.uiServer.stop();
		} 
        catch (Exception e) 
        {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println("*** 保存模型失败！！！ ***");
		}
	}
	
	public int getIndex() {
		return index;
	}
	
	public YoloConfig getYOLO_CONFIG() {
		return YOLO_CONFIG;
	}
	public void setYOLO_CONFIG(YoloConfig yOLO_CONFIG) {
		if(yOLO_CONFIG != null)
		{
			this.YOLO_CONFIG = yOLO_CONFIG;
		}
	}
	public String getDATA_DIR() {
		return DATA_DIR;
	}
	public void setDATA_DIR(String dATA_DIR) {
		DATA_DIR = dATA_DIR;
	}

	@Override
	public void run() {
		// TODO Auto-generated method stub
		this.train(this.YOLO_CONFIG, this.DATA_DIR);
		this.savemodel(this.modelpath);
	}
	
}
