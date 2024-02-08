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

int main()
{
    
    Mat background_ref = imread("../../outputDataset/FLOOR_Y-F1-B0_P220533_20201203141353_275/FLOOR_Y-F1-B0_P220533_20201203141353_275_cs010_00372.png", IMREAD_GRAYSCALE);
    std::string path = "../../dataset/";
    std::vector<std::string> dataset_path;

    std::set<std::filesystem::path> sorted_by_name;

    int nb_img = 0;

    double alpha = 0.5; double beta = 0.5;
    Mat avg_img = imread("../../outputDataset/FLOOR_Y-F1-B0_P220533_20201203141353_275/FLOOR_Y-F1-B0_P220533_20201203141353_275_cs010_00372.png", IMREAD_GRAYSCALE);
                if(avg_img.empty())
            {
                std::cout << "Could not read the image: avg img" << std::endl;
                return 1;
            }
    for (const auto & entry : std::filesystem::directory_iterator{path}){
        std::string dirname = entry.path().filename();

        if (dirname != "FLOOR_Y-F1-B0_P220533_20201203141353_275") {
            continue;
        }
        for (const auto & file_entry : std::filesystem::directory_iterator{entry.path()}){
            // Sort filepath
            sorted_by_name.insert(file_entry.path());

            // Get Average
            std::string image_path = samples::findFile(file_entry.path());
            Mat img = imread(image_path, IMREAD_GRAYSCALE);
            if(img.empty())
            {
                std::cout << "Could not read the image: " << image_path << std::endl;
                return 1;
            }
            avg_img += img;
            nb_img++;
        }
    }
    avg_img.convertTo(avg_img, CV_8U, 1.0/nb_img); 
std::cout << "nb img : "<< nb_img << std::endl;


    avg_img = avg_img/nb_img;
    imshow("AVERAGE IMG ", avg_img);
    waitKey(0);

  //int erosion_type = 0;
  //if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  //else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  //else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
  int  erosion_size = 30;
  Mat element = getStructuringElement( MORPH_RECT,
                       Size( 2*erosion_size + 1, 2*erosion_size+1 ));
  erode( avg_img, avg_img, element );

    for (const auto & entry : std::filesystem::directory_iterator{path}){
        std::string dirname = entry.path().filename();

        if (dirname != "FLOOR_Y-F1-B0_P220533_20201203141353_275") {
            std::cout << "SKipped : "<< dirname << std::endl;

            continue;
        }
        
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
            Mat dst_diff = dst - background_ref;
            std::string filename = path.filename();
            imshow("Display window", dst_diff);
            int k = waitKey(0); // Wait for a keystroke in the window
            if(k == 'a'){
                return 0;
            }
            // if(k == 's')
            // {
            //imwrite("../../outputDataset/" + dirname + "/" + filename, dst);
            // std::cout << "../../outputDataset/" + dirname + "/" + filename << std::endl;
            // }
        }
    }
    return 0;
}