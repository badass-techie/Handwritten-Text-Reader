#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <utility>

//MNIST dataset
//
//Object contains pixel values for all images in the dataset, and the corresponding labels
class MNIST{
    private:
        std::ifstream image_file, label_file;                                   //file streams
        uint32_t magic, num_items, num_labels, rows, cols;                      //metadata
        char* memblock;                                                         //memory block for i/o
        std::vector <std::pair <std::vector <double>, std::vector <double>>> images_and_labels;     //vector of images and corresponding labels
        std::vector <std::vector <double>> images;                              //vector of images
        std::vector <std::vector <double>> labels;                              //vector of labels

        //swap endian
        uint32_t swap_endian(uint32_t val);
    public:
        //constructor
        MNIST(std::string img_dir, std::string lbl_dir, int no_of_images, int first_image = 1);

        //returns a vector of images and corresponding labels
        std::vector <std::pair <std::vector <double>, std::vector <double>>>& get_images_and_labels();
};