#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>
#include <string>
using namespace cv;

int main()
{

    std::string path = "../../dataset/";
    std::vector<std::string> dataset_path;
    for (const auto & entry : std::filesystem::directory_iterator{path}){
        std::string dirname = entry.path().filename();
        for (const auto & file_entry : std::filesystem::directory_iterator{entry.path()}){
            std::filesystem::path path = file_entry.path();
            // std::cout << path << std::endl;

            std::string image_path = samples::findFile(path);
            Mat img = imread(image_path, IMREAD_GRAYSCALE);
            if(img.empty())
            {
                std::cout << "Could not read the image: " << image_path << std::endl;
                return 1;
            }
            Mat dst;
            equalizeHist(img, dst);
            std::string filename = path.filename();
            imshow("Display window", dst);
            // int k = waitKey(0); // Wait for a keystroke in the window
            // if(k == 's')
            // {
            imwrite("../../outputDataset/" + dirname + "/" + filename, dst);
            // std::cout << "../../outputDataset/" + dirname + "/" + filename << std::endl;
            // }
        }
    }
    return 0;
}