#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <exception>
#include <fstream>
#include <string>
#include "../include/MNIST.h"

using uchar = unsigned char;

uint32_t MNIST::swap_endian(uint32_t val){
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

MNIST::MNIST(std::string img_dir, std::string lbl_dir, int no_of_images, int first_image){
    try{
        image_file.open(img_dir, std::ios::binary);
        label_file.open(lbl_dir, std::ios::binary);
        //metadata
        image_file.read(reinterpret_cast<char*>(&magic), 4);
        magic = swap_endian(magic);
        if(magic != 2051) throw std::string("Incorrect image file magic");

        label_file.read(reinterpret_cast<char*>(&magic), 4);
        magic = swap_endian(magic);
        if(magic != 2049) throw std::string("Incorrect label file magic");

        image_file.read(reinterpret_cast<char*>(&num_items), 4);
        num_items = swap_endian(num_items);
        if(first_image + no_of_images > num_items + 1) throw std::string("no_of_images after first_image pass EOF. File has only " + std::to_string(num_items) + " images");
        label_file.read(reinterpret_cast<char*>(&num_labels), 4);
        num_labels = swap_endian(num_labels);
        if(num_items != num_labels) throw std::string("Image file nums should be equal to label nums");

        image_file.read(reinterpret_cast<char*>(&rows), 4);
        rows = swap_endian(rows);
        image_file.read(reinterpret_cast<char*>(&cols), 4);
        cols = swap_endian(cols);

        //move get positions
        image_file.seekg((first_image - 1) * rows * cols, std::ios::cur);
        label_file.seekg(first_image - 1, std::ios::cur);
        for(int img_no = 0; img_no < no_of_images; ++img_no){
            std::pair <std::vector <double>, std::vector <double>> image_and_label;
            //read pixel data
            memblock = new char[rows * cols];
            image_file.read(memblock, rows * cols);
            std::vector <double> current_image; //vector of all pixel values for the image
            std::for_each(memblock, memblock + rows * cols, [&current_image](char& character){
                current_image.push_back((double)(uchar)character/255);
            });
            image_and_label.first = current_image;
            delete[] memblock;
            
            //read label
            char lbl;
            label_file.read(&lbl, 1);
            std::vector <double> current_label; //vector representing a label in this manner, 3 - {0, 0, 0, 1, 0, 0, 0, 0, 0, 0} - you get the idea
            for(int index = 0; index < 10; ++index) current_label.push_back((uint32_t)lbl == index); //sets the
                //element of the vector to 1 if the index matches that of the label and 0 if it does not
            image_and_label.second = current_label;
            images_and_labels.push_back(image_and_label);
        }
        image_file.close();
        label_file.close();
    }
    catch (std::string exc){
        std::cout<<"Error: "<<exc<<std::endl;
    }
    catch (std::bad_alloc& ba){
        std::cout<<"Error: "<<ba.what()<<std::endl;
    }
    catch (std::exception& e){
        std::cout<<"Error: "<<e.what()<<std::endl;
    }
}

std::vector <std::pair <std::vector <double>, std::vector <double>>>& MNIST::get_images_and_labels(){
    return images_and_labels;
}