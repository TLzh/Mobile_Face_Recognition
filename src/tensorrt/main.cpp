#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <algorithm>
#include <sys/stat.h>
#include <ctime>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"
using namespace nvinfer1;
using namespace nvcaffeparser1;

static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int INPUT_C = 3;
static const int OUTPUT_SIZE = 512;
static Logger gLogger;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "embedding";
const std::vector<std::string> net_dir{"network/"};
const std::vector<std::string> img_dir{"image/"};


//void readJPEGFile(const std::string& fileName, uint8_t buffer[INPUT_H*INPUT_W])
//{
//    readJPGFile(fileName, buffer, INPUT_H, INPUT_W);
//}

void caffeToGIEModel(
        const std::string& deployFile, 
        const std::string& modelFile, 
        const std::vector<std::string>& outputs,
        unsigned int maxBatchSize,
        IHostMemory *&gieModelStream)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    const IBlobNameToTensor* blobNameToTensor = parser->parse(
               locateFile(deployFile, net_dir).c_str(),
               locateFile(modelFile, net_dir).c_str(),
               *network,
               DataType::kFLOAT);

    // specify which tensors are outputs
    for (auto& s : outputs)
            network->markOutput(*blobNameToTensor->find(s.c_str()));
    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    gieModelStream = engine->serialize();
    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}
void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
        const ICudaEngine& engine = context.getEngine();
        // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
        // of these, but in this case we know that there is exactly one input and one output.
        assert(engine.getNbBindings() == 2);
        void* buffers[2];

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // note that indices are guaranteed to be less than IEngine::getNbBindings()
        int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
                outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

        // create GPU buffers and a stream
        CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // release the stream and the buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));
}
int main(int argc, char** argv)
{
    // create a GIE model from the caffe model and serialize it to a stream
    IHostMemory *gieModelStream{nullptr};
    caffeToGIEModel("mobilenet_v2_deploy.prototxt", "solver_iter_100000.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, gieModelStream);
    
    
    std::string imgname1 = locateFile(argv[1], img_dir);
    std::string imgname2 = locateFile(argv[2], img_dir);
    //std::string imgname3 = locateFile(argv[3], img_dir);
    cv::Mat inimg1 = cv::imread(imgname1);
    cv::Mat inimg2 = cv::imread(imgname2);
    //cv::Mat inimg3 = cv::imread(imgname3);
    cv::Size dsize(INPUT_W, INPUT_H);
    cv::Mat image1 = cv::Mat(dsize, CV_8UC3);
    cv::Mat image2 = cv::Mat(dsize, CV_8UC3);
    //cv::Mat image3 = cv::Mat(dsize, CV_8UC3);
    cv::resize(inimg1, image1, dsize);
    cv::resize(inimg2, image2, dsize);
    //cv::resize(inimg3, image3, dsize);
    

    float data1[INPUT_C*INPUT_H*INPUT_W];
    float data2[INPUT_C*INPUT_H*INPUT_W];
    for (int i = 0, volChl = INPUT_H * INPUT_W; i < INPUT_C; i++)
    {
        for (int j = 0; j < volChl; j++)
        {
            data1[i*volChl + j] = float(*(image1.ptr<uchar>(0) + j*INPUT_C + i));
            data2[i*volChl + j] = float(*(image2.ptr<uchar>(0) + j*INPUT_C + i));
        }
    }
    //float data2[INPUT_C*INPUT_H*INPUT_w];
    //for (int i = 0; i < INPUT_H*INPUT_W*INPUT_C; i++)
    //{
    //    data2[i] = float(*(image2.ptr<uchar>(0)+i));
    //}
    //float data3[INPUT_C*INPUT_W*INPUT_H];
    //for (int i = 0; i < INPUT_H*INPUT_W*INPUT_C; i++)
    //{
    //    data3[i] = float(*(image3.ptr<uchar>(0)+i));
    //}
    
    
    // deserialize the engine 
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), nullptr);
    if (gieModelStream) gieModelStream->destroy();

    IExecutionContext *context = engine->createExecutionContext();

    // run inference
    float embedding1[OUTPUT_SIZE], embedding2[OUTPUT_SIZE];//, embedding3[OUTPUT_SIZE];
    
    std::cout << "Start do inference." << std::endl;
    clock_t start, end;
    double cost = 0;
    
    
    for (int i = 0; i < 1; i++)
    {
        start = clock();
        doInference(*context, data1, embedding1, 1);
        doInference(*context, data2, embedding2, 1);
    //doInference(*context, data3, embedding3, 1);
        end = clock();
        cost += double(end - start);
    }
    
    
    cost = cost / CLOCKS_PER_SEC;
    std::cout << "Do inference cost " << cost << "s." << std::endl; 


    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    
    float dist1 = 0;
    //float dis2 = 0;
    
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        dist1 += (embedding1[i] - embedding2[i]) * (embedding1[i] - embedding2[i]);
        //dis2 += (embedding1[i] - embedding3[i]) * (embedding1[i] - embedding3[i]);
    }
    std::cout << dist1;
    std::cout << std::endl;
    //std::cout << dis2 << std::endl;
    
    return 0;
}

