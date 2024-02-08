#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>
#include <string>
#include <set>
#include <regex>
using namespace cv;
namespace fs = std::filesystem;

// This program is used to denormalize all the images

int main()
{

    std::string path = "../../dataset/";
    std::vector<std::string> dataset_path;
    std::set<fs::path> sorted_by_name;
    for (const auto & entry : std::filesystem::directory_iterator{path}){
        std::string dirname = entry.path().filename();
        fs::create_directories("../../outputDataset/" + dirname);
        for (const auto & file_entry : std::filesystem::directory_iterator{entry.path()}){
            sorted_by_name.emplace(file_entry.path());
        }
    }

    std::regex pattern("dataset");

    for (const auto filepath : sorted_by_name){
        // std::filesystem::path path = file_entry.path();
        // std::cout << path << std::endl;

        std::string image_path = samples::findFile(filepath);
        Mat img = imread(image_path, IMREAD_GRAYSCALE);
        if(img.empty())
        {
            std::cout << "Could not read the image: " << image_path << std::endl;
            return 1;
        }
        Mat dst;
        equalizeHist(img, dst);
        std::string filename = filepath.filename();
        imshow("Display window", dst);
        int k = waitKey(0); // Wait for a keystroke in the window
        
        imwrite(std::regex_replace(image_path, pattern, "outputDataset"), dst);

    }
    return 0;
}