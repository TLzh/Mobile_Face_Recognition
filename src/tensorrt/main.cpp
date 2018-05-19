#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <algorithm>
#include <sys/stat.h>
#include <time.h>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"
using namespace nvinfer1;
using namespace nvcaffeparser1;

static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int INPUT_C = 3;
static const int OUTPUT_SIZE = 1000;
static Logger gLogger;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
const std::vector<std::string> directories{"caffe_model/"};
std::string locateFile(const std::string& input)
{
    return locateFile(input, directories);
}

void readJPEGFile(const std::string& fileName, uint8_t buffer[INPUT_H*INPUT_W])
{
//    readJPGFile(fileName, buffer, INPUT_H, INPUT_W);
}

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
               locateFile(deployFile, directories).c_str(),
               locateFile(modelFile, directories).c_str(),
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
        CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
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
    int kk = 231;
    int xx[10];
    std::cout << xx[10] << std::endl;
    // create a GIE model from the caffe model and serialize it to a stream
    IHostMemory *gieModelStream{nullptr};
    caffeToGIEModel("mobilenet_v2_deploy.prototxt", "mobilenet_v2.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, gieModelStream);

    // read a random digit file
    
    //srand(unsigned(time(nullptr)));
    uint8_t fileData[INPUT_H*INPUT_W];
    //int num = rand() % 10; 
    //readPGMFile(locateFile("cat.jpg", directories), fileData);
    std::string imgname = locateFile("cat.jpg", directories);
    cv::Mat inimg = cv::imread(imgname);
    const float img_mean[3] = {103.94, 116.78, 123.68};
    cv::Size dsize(INPUT_W, INPUT_H);
    cv::Mat image = cv::Mat(dsize, CV_8UC3);
    cv::resize(inimg, image, dsize);
    // parse the mean file and      subtract it from the image
    //ICaffeParser* parser = createCaffeParser();
    //IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(locateFile("mnist_mean.binaryproto", directories).c_str());
    //parser->destroy();

    //const float *meanData = reinterpret_cast<const float*>(meanBlob->getData());

    float data[INPUT_H*INPUT_W*INPUT_C];
    for (int i = 0; i < INPUT_H*INPUT_W*INPUT_C; i++)
        data[i] = float(*(image.ptr<uchar>(0)+i));

    //meanBlob->destroy();

    // deserialize the engine 
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), nullptr);
    if (gieModelStream) gieModelStream->destroy();

    IExecutionContext *context = engine->createExecutionContext();

    // run inference
    float prob[1000];
    doInference(*context, data, prob, 1);

    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    float max_value = 0;
    int max_index = -1;
    for (int i = 0; i < 1010; i++)
    {
        std::cout << prob[i] << ",";
        if (prob[i] > max_value)
        {
            max_value = prob[i];
            max_index = i;
        }
    }
    std::cout << std::endl;
    std::cout << max_value << std::endl;
    std::cout << max_index << std::endl;
/*
    // print a histogram of the output distribution
    std::cout << "\n\n";
    float val{0.0f};
    int idx{0};
    for (unsigned int i = 0; i < 10; i++)
    {
        val = std::max(val, prob[i]);
        if (val == prob[i]) idx = i;
            std::cout << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
    }
    std::cout << std::endl;

    return (idx == num && val > 0.9f) ? EXIT_SUCCESS : EXIT_FAILURE;
*/
}

