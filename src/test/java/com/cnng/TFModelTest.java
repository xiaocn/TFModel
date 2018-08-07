package com.cnng;

import com.google.common.io.ByteStreams;
import org.apache.log4j.Logger;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import static org.junit.Assert.*;

/**
 * @Version : v1.0
 * @ClassName : TFModelTest
 * @PackageName : com.cnng
 * @ProjectName : TFModelInter
 * @Description : 测试tensorflow的模型接口
 * @Author : lixiao
 * @Date : 2018-08-07下午8:47
 * @modified by
 */
public class TFModelTest {

    private Logger log = Logger.getLogger(TFModel.class); //定义日志

    @Before
    public void setUp() {
        log.info("---------------开始测试方法---------------------");
    }

    @After
    public void tearDown() {
        log.info("---------------方法测试完毕---------------------");
    }

    @Test
    public void loadTxtLabels() {
        String basePath = "./src/test/resources/test_data_loadLabel";
        String[] path = {null,"",basePath+"/中文/stand.txt",basePath+ "/en/empty",basePath+ "/en/stand.txt",basePath+ "/en/other"};
        String[][] answer = {{}, {},{"a","b","c"},{},{"a","b","c","中文"},{"g","f","g"} };
        long start = System.currentTimeMillis();
        for(int i = 0; i<path.length; i++){
            long each_time = System.currentTimeMillis();
            List<String> labels = new ArrayList<>();
            try {
               labels = TFModel.loadTxtLabels(path[i]);
                assertArrayEquals("数组不相等，加载标签接口错误",answer[i],labels.toArray());
                log.info("标签文件加载成功");
            } catch (Exception e) {
                assertArrayEquals("数组不相等，加载标签接口错误",answer[i],labels.toArray());
            }
            log.info("--"+i+"--测试时间为："+(System.currentTimeMillis()-each_time)+"ms");
        }
        log.info("平均测试时间为："+(System.currentTimeMillis()-start) / path.length +"ms");
    }

    @Test
    public void loadModel() {
        String basePath = "./src/test/resources/test_data_loadModel";
        String[] path = {null,"",basePath+"/中文/color_fc2_model.pb",basePath+ "/en/empty.pb",basePath+"/en/txt.pb",basePath+ "/en/mpp_detection_model.pb",basePath+ "/en/clear_muddy_model"};
        long start = System.currentTimeMillis();
        for(int i = 0; i<path.length; i++){
            long each_time = System.currentTimeMillis();
            byte[] data = null;
            try {
                data = TFModel.loadModel(path[i]);
                assertNotNull(data);
                log.info("模型文件加载成功");
            } catch (Exception e) {
                assertNull("加载模型接口错误",data);
                log.info("模型文件没有加载成功");
            }
            log.info("--"+i+"--测试时间为："+(System.currentTimeMillis()-each_time)+"ms");
        }
        log.info("平均测试时间为："+(System.currentTimeMillis()-start) / path.length +"ms");
    }

    @Test
    public void classifyImage() {
        String basePath = "./src/test/resources/test_data_classify";
        String[] modelPath = {null,basePath+"/color_fc2_model.pb"};
        String[] imagePath = {null, basePath+"/0.jpg"};
        String[] labelPath = {null, basePath+"/color_fc2_label.txt"};
        String[] answers = {"purple","red"};
        long start = System.currentTimeMillis();
        for(int i = 0; i<modelPath.length; i++){
            byte[] model;
            byte[] data;
            List<String> labels;
            try (InputStream is =new FileInputStream(new File(Objects.requireNonNull(imagePath[i])).getPath())){
                model = TFModel.loadModel(modelPath[i]);
                data = ByteStreams.toByteArray(is);
                labels = TFModel.loadTxtLabels(labelPath[i]);
            } catch (Exception e) {
                log.info("model is null and img is null");
                continue;
            }
            long each_time = System.currentTimeMillis();
            int label;
            try (Graph g = new Graph(); Session sess = new Session(g)){
                g.importGraphDef(model != null ? model : new byte[0]);
                label = TFModel.classifyImage(data,sess,"input","output");
                assertEquals("分类接口错误",labels.get(label),answers[i]);
                log.info("分类结果： "+labels.get(label));
            }
            log.info("--"+i+"--测试时间为："+(System.currentTimeMillis()-each_time)+"ms");
        }
        log.info("平均测试时间为："+(System.currentTimeMillis()-start) / modelPath.length +"ms");
    }

    @Test
    public void detectionImage() {
        String basePath = "./src/test/resources/test_data_detection";
        String[] modelPath = {null,basePath+"/mpp_detection_model.pb"};
        String[] imagePath = {null, basePath+"/microplate.JPG"};
        String[] labelPath = {null, basePath+"/mpp_detection_label.txt"};
        String[] answer = {null, "microplate"};
        long start = System.currentTimeMillis();
        for(int i = 0; i<modelPath.length; i++){
            byte[] model;
            Tensor<UInt8> input;
            List<String> labels;
            try (InputStream is =new FileInputStream(new File(Objects.requireNonNull(imagePath[i])).getPath())){
                model = TFModel.loadModel(modelPath[i]);
                BufferedImage image = ImageIO.read(is);
                byte[] data = ByteStreams.toByteArray(is);
                input = Tensor.create(UInt8.class,new long[]{1,image.getHeight(),image.getWidth(),3},ByteBuffer.wrap(data));
                labels = TFModel.loadTxtLabels(labelPath[i]);
            } catch (Exception e) {
                log.info("model is null and img is null");
                continue;
            }
            long each_time = System.currentTimeMillis();
            try (Graph g = new Graph(); Session sess = new Session(g)){
                g.importGraphDef(model != null ? model : new byte[0]);
                List<float[]> results = TFModel.detectionImage(input, sess, 0.9f, "image_tensor",
                        "detection_boxes", "detection_scores", "detection_classes");
                assertEquals("检测接口错误",labels.get((int)results.get(0)[1]),answer[i]);
                for(float[] result: results) {
                    log.info("检测结果：" + Arrays.toString(result));
                }
            }
            log.info("--"+i+"--测试时间为："+(System.currentTimeMillis()-each_time)+"ms");
        }
        log.info("平均测试时间为："+(System.currentTimeMillis()-start) / modelPath.length +"ms");
    }
}