package com.cnng;

import com.google.common.io.ByteStreams;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * @Version : v1.0
 * @ClassName : TFModel
 * @PackageName : com.cnng
 * @ProjectName : TFModelInter
 * @Description : 实现java调用tensorflow的模型文件，给后台调用人工只能算法
 * @Author : lixiao
 * @Date : 2018-08-07下午7:15
 * @modified by
 */
public class TFModel {

    /**
     * @MethodName  : loadTxtLabels
     * @ClassName   : TFModel
     * @Description : 加载标签文件
     * @Author      : lixiao
     * @Date        : 18-8-7 下午7:21
     * @Param: labelPath 标签文件的路径，string类型
     * @return java.util.List<java.lang.String> 列表
     * @throw
     * @Modeify by
     */
    public static List<String> loadTxtLabels(String labelPath) throws IOException {
        int initSize = 1000;
        ArrayList<String> labels = new ArrayList<>(initSize);
        String line;
        try (InputStream is = new FileInputStream(labelPath);
             BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }
        }
        return labels;
    }

    /**
     * @MethodName  : loadModel
     * @ClassName   : TFModel
     * @Description : 加载tensorflow-pb模型文件
     * @Author      : lixiao
     * @Date        : 18-8-7 下午7:58
     * @Param: pbModelPath pb模型文件路径
     * @return byte[] 返回的pb文件字节码
     * @throw
     * @Modeify by
     */
    public static byte[] loadModel(String pbModelPath) throws IOException {
        try (InputStream is =new FileInputStream(new File(pbModelPath).getPath())) {
            return ByteStreams.toByteArray(is);
        }
    }

    /**
     * @MethodName  : classifyImage
     * @ClassName   : TFModel
     * @Description : tensorflow的分类模型
     * @Author      : lixiao
     * @Date        : 18-8-7 下午8:43
     * @Param: image 输入图像的字节码
     * @Param: model 加载模型的session
     * @Param: inputName 输入节点的名称
     * @Param: outputName 输出节点的名称
     * @return int 返回所属类别的下标
     * @throw
     * @Modeify by
     */
    public static int classifyImage(byte[] image, Session model, String inputName, String outputName){
        Tensor<?> input =  Tensor.create(image);
        try (Tensor<Float> results = model.runner()
                .feed(inputName, input)
                .fetch(outputName)
                .run().get(0).expect(Float.class)) {
            float[] scoreT = results.copyTo(new float[1][(int) (results.shape()[1])])[0];
            int labelIndex = 0;
            float score = scoreT[0];
            for (int i = 1; i < scoreT.length; i++) {
                if (scoreT[i] > score) {
                    labelIndex = i;
                    score = scoreT[i];
                }
            }
            return labelIndex;
        }
    }

    /**
     * @MethodName  : detectionImage
     * @ClassName   : TFModel
     * @Description : tensorflow物体检测跟踪模型
     * @Author      : lixiao
     * @Date        : 18-8-7 下午8:36
     * @Param: input 输入的图像数据
     * @Param: model 加载模型的session
     * @Param: rate  预计的准确率
     * @Param: inputName 输入节点的名称
     * @Param: boxesName 输出节点的框名称
     * @Param: scoreName 输出节点的分数名称
     * @Param: className 输出节点的分类名称
     * @return java.util.List<float[]> 返回检测到物体的各个结果，每个物体的信息总共有六个，分别为分数，物体名称的下标，矩形框的左上点X，y坐标，矩形框右下角的x，y坐标（坐标为相应长度的比例）
     * @throw
     * @Modeify by
     */
    public static List<float[]> detectionImage(Tensor<UInt8> input, Session model, float rate,
                                        String inputName, String boxesName, String scoreName, String className){
        List<Tensor<?>> outputs = model.runner()
                .feed(inputName, input)
                .fetch(boxesName)
                .fetch(scoreName)
                .fetch(className)
                .run();
        try (Tensor<Float> boxesT = outputs.get(0).expect(Float.class);
             Tensor<Float> scoresT = outputs.get(1).expect(Float.class);
             Tensor<Float> classesT = outputs.get(2).expect(Float.class)) {
            float[][] b = boxesT.copyTo(new float[1][(int) (boxesT.shape()[1])][4])[0];
            float[] s = scoresT.copyTo(new float[1][(int) (scoresT.shape()[1])])[0];
            float[] c = classesT.copyTo(new float[1][(int) (classesT.shape()[1])])[0];
            List<float[]> results = new ArrayList<>();
            for (int i = 0; i < b.length; i++) {
                if (s[i] > rate) {
                    results.add(new float[]{s[i], c[i], b[i][1], b[i][0], b[i][3], b[i][2]});
                }
            }
            return results;
        }
    }

}
