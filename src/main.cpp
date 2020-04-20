#include <iostream>
#include <vector>
#include <chrono>
#include "../EasyBMP_1.06/EasyBMP.h"
#include "../include/MNIST.h"
#include "../include/NeuralNetwork.h"

using std::cin;
using std::cout;
using std::endl;
using std::string;
using namespace std::chrono;

using uint = unsigned int;

//extracts images from the MNIST test set into the specified location
void generate_samples(string folder, int num_of_images = 10000){
    if (num_of_images > 10000) num_of_images = 10000;
    string basedir = "../data/MNIST_dataset/", imgdir = basedir + "t10k-images.idx3-ubyte", lbldir = basedir + "t10k-labels.idx1-ubyte";
    BMP img;
    img.SetSize(28, 28);
    img.SetBitDepth(32);
    MNIST* images = new MNIST(imgdir, lbldir, num_of_images);
    for(auto& image: images->get_images_and_labels()){
        int numimg = &image - &images->get_images_and_labels()[0] + 1;
        for(int row = 0; row < 28; ++row){
            for(int pixel = 0; pixel < 28; ++pixel){
                img(pixel, row)->Red = 255 - uint32_t(image.first[28*row + pixel] * 255);
                img(pixel, row)->Green = 255 - uint32_t(image.first[28*row + pixel] * 255);
                img(pixel, row)->Blue = 255 - uint32_t(image.first[28*row + pixel] * 255);
                img(pixel, row)->Alpha = 0;
            }
        }
        string filename = folder + "sample" + std::to_string(numimg) + ".bmp";
        img.WriteToFile(filename.c_str());
        cout<<"Sample "<<numimg<<" written successfully";
        for(double& i: image.second) if (i) cout<<" (label is "<<&i - &image.second[0]<<")"<<endl;
    }
    delete images;
}

//trains the network
void train(){
    milliseconds time_elapsed = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    string base_dir = "../data/MNIST_dataset/";
    string train_images_dir = base_dir + "train-images.idx3-ubyte", train_labels_dir = base_dir + "train-labels.idx1-ubyte";
    string test_images_dir = base_dir + "t10k-images.idx3-ubyte", test_labels_dir = base_dir + "t10k-labels.idx1-ubyte";
    MNIST* training_set = new MNIST(train_images_dir, train_labels_dir, 60000);
    MNIST* test_set = new MNIST(test_images_dir, test_labels_dir, 10000);
    NeuralNetwork* network = new NeuralNetwork({784, 28, 10});
    double eta;
    int epochs;
    string filename;
    cout<<"Enter the learning rate>";
    cin>>eta;
    cout<<"Enter the number of epochs to train for>";
    cin>>epochs;
    cout<<"Enter the name of the file to write the weights and biases to at the end of training>";
    cin>>filename;
    network->train(training_set->get_images_and_labels(), epochs, 6, eta, test_set->get_images_and_labels(), filename);
    delete network;
    delete test_set;
    delete training_set;
    time_elapsed = duration_cast<milliseconds>(system_clock::now().time_since_epoch()) - time_elapsed;
    cout<<endl<<"Time elapsed: "<<static_cast<double>(time_elapsed.count())/1000<<" second(s)"<<endl;
}

//tests the network
void test(const char* filename){
    cout<<"Program that recognizes handwritten digits";
    NeuralNetwork* network = new NeuralNetwork(filename);
    BMP image;
    string token;
    start:
    cout<<endl<<"Enter the relative path of the image>";
    cin>>token;
    image.ReadFromFile(token.c_str());
    std::vector <double> pixels;
    for(int row = 0; row < 28; ++row){
        for(int pixel = 0; pixel < 28; ++pixel){
            pixels.push_back((double(uint(image(pixel, row)->Red))/255 + double(uint(image(pixel, row)->Green))/255 + double(uint(image(pixel, row)->Blue))/255)/3);
        }
    }
    std::vector <string> digits{"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
    cout<<endl<<"Hmm... looks like a "<<digits[network->feed_forward(pixels)]<<"!"<<endl<<endl;
    cout<<"Enter restart to restart, or anything else to quit>";
    cin>>token;
    if(token == "restart") goto start;
    delete network;
}

int main(){
    test("../data/weights_and_biases.csv");
    //train();
    //generate_samples("../data/sample_images_2/");
    return 0;
}