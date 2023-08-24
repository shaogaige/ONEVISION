/**
 * ClassName:YoloConfig.java
 * Date:2022年4月8日
 */
package com.idata.vision;

import com.google.gson.Gson;

/**
 * Creater:SHAO Gaige
 * Description:yolo模型配置参数
 * Log:
 */
public class YoloConfig {
	//类别名称
	public String CLASSNAME = "大疆M300";
	private String[] CLASSESNAME = {"大疆M300"};
	//类别个数
	public int CLASSES_NUMBER = 1;
	//输入图像宽度
	public int INPUT_WIDTH = 416;
	//输入图像高度
	public int INPUT_HEIGHT = 416;
	//颜色通道
	public int CHANNELS = 3;
    //格网宽度
	public int GRID_WIDTH = 13;
	//格网高度
	public int GRID_HEIGHT = 13;
	//预测个数
	public int BOXES_NUMBER = 5;
	//预测范围
	public double[][] PRIOR_BOXES = {{1.5, 1.5}, {2, 2}, {3, 3}, {3.5, 8}, {4, 9}};
	//最小批量
	public int BATCH_SIZE = 4;
	//循环次数
	public int EPOCHS = 50;
	//学习速率
	public double LEARNIGN_RATE = 0.0001;
	//随机种子
	public int SEED = 7854;
    //数据集路径
	/*parent Dataset folder "DATA_DIR" contains two subfolder "images" and "annotations" */
	public String DATA_DIR = "";
	//配置文件路径
	public String CONFIG_PATH = "";
	//模型文件路径
	public String MODEL_PATH = "";
	//类别文件路径
	public String CLASS_PATH = "";
	/* Yolo loss function prameters for more info
	   https://stats.stackexchange.com/questions/287486/yolo-loss-function-explanation*/
	public double LAMDBA_COORD = 1.0;
	public double LAMDBA_NO_OBJECT = 0.5;
	
	
	public String getCLASSNAME() {
		return CLASSNAME;
	}
	public void setCLASSNAME(String cLASSNAME) {
		CLASSNAME = cLASSNAME;
	}
	public String getCLASSNAME_CH() {
		return CLASSNAME;
	}
	public void setCLASSNAME_CH(String cLASSNAME_CH) {
		CLASSNAME = cLASSNAME_CH;
	}
	public int getCLASSES_NUMBER() {
		return CLASSES_NUMBER;
	}
	public void setCLASSES_NUMBER(int cLASSES_NUMBER) {
		CLASSES_NUMBER = cLASSES_NUMBER;
	}
	public String getCONFIG_PATH() {
		return CONFIG_PATH;
	}
	public void setCONFIG_PATH(String cONFIG_PATH) {
		CONFIG_PATH = cONFIG_PATH;
	}
	public String getMODEL_PATH() {
		return MODEL_PATH;
	}
	public void setMODEL_PATH(String mODEL_PATH) {
		MODEL_PATH = mODEL_PATH;
	}
	public int getINPUT_WIDTH() {
		return INPUT_WIDTH;
	}
	public void setINPUT_WIDTH(int iNPUT_WIDTH) {
		INPUT_WIDTH = iNPUT_WIDTH;
	}
	public int getINPUT_HEIGHT() {
		return INPUT_HEIGHT;
	}
	public void setINPUT_HEIGHT(int iNPUT_HEIGHT) {
		INPUT_HEIGHT = iNPUT_HEIGHT;
	}
	public int getCHANNELS() {
		return CHANNELS;
	}
	public void setCHANNELS(int cHANNELS) {
		CHANNELS = cHANNELS;
	}
	public int getGRID_WIDTH() {
		return GRID_WIDTH;
	}
	public void setGRID_WIDTH(int gRID_WIDTH) {
		GRID_WIDTH = gRID_WIDTH;
	}
	public int getGRID_HEIGHT() {
		return GRID_HEIGHT;
	}
	public void setGRID_HEIGHT(int gRID_HEIGHT) {
		GRID_HEIGHT = gRID_HEIGHT;
	}
	public int getBOXES_NUMBER() {
		return BOXES_NUMBER;
	}
	public void setBOXES_NUMBER(int bOXES_NUMBER) {
		BOXES_NUMBER = bOXES_NUMBER;
	}
	public double[][] getPRIOR_BOXES() {
		return PRIOR_BOXES;
	}
	public void setPRIOR_BOXES(double[][] pRIOR_BOXES) {
		PRIOR_BOXES = pRIOR_BOXES;
	}
	public int getBATCH_SIZE() {
		return BATCH_SIZE;
	}
	public void setBATCH_SIZE(int bATCH_SIZE) {
		BATCH_SIZE = bATCH_SIZE;
	}
	public int getEPOCHS() {
		return EPOCHS;
	}
	public void setEPOCHS(int ePOCHS) {
		EPOCHS = ePOCHS;
	}
	public double getLEARNIGN_RATE() {
		return LEARNIGN_RATE;
	}
	public void setLEARNIGN_RATE(double lEARNIGN_RATE) {
		LEARNIGN_RATE = lEARNIGN_RATE;
	}
	public int getSEED() {
		return SEED;
	}
	public void setSEED(int sEED) {
		SEED = sEED;
	}
	public String getDATA_DIR() {
		return DATA_DIR;
	}
	public void setDATA_DIR(String dATA_DIR) {
		DATA_DIR = dATA_DIR;
	}
	public double getLAMDBA_COORD() {
		return LAMDBA_COORD;
	}
	public void setLAMDBA_COORD(double lAMDBA_COORD) {
		LAMDBA_COORD = lAMDBA_COORD;
	}
	public double getLAMDBA_NO_OBJECT() {
		return LAMDBA_NO_OBJECT;
	}
	public void setLAMDBA_NO_OBJECT(double lAMDBA_NO_OBJECT) {
		LAMDBA_NO_OBJECT = lAMDBA_NO_OBJECT;
	}
	
	
	public String tojson()
	{
		return new Gson().toJson(this);
	}
	
	public static YoloConfig fromjson(String jsonstr)
	{
		return new Gson().fromJson(jsonstr, YoloConfig.class);
	}

	public String[] getCLASSESNAME() {
		if(CLASSESNAME == null)
		{
			CLASSESNAME = this.CLASSNAME.split(",");
		}
		return CLASSESNAME;
	}
	

}
