#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <filesystem>
#include <string>
#include <set>
#include <regex>
#include <opencv2/objdetect.hpp>
using namespace cv;
namespace fs = std::filesystem;

// This program is used to denormalize all the images

int main()
{

    Ptr<BackgroundSubtractor> pBackSub;

    Ptr<BackgroundSubtractor> pBackSub2;
    std::string path = "../../dataset/";
    std::vector<std::string> dataset_path;
    for (const auto & entry : std::filesystem::directory_iterator{path}){

        std::set<fs::path> sorted_by_name;
        std::string dirname = entry.path().filename();
        fs::create_directories("../../outputDataset/" + dirname);
        for (const auto & file_entry : std::filesystem::directory_iterator{entry.path()}){
            sorted_by_name.emplace(file_entry.path());
        }
        std::regex pattern("dataset");
        Mat fgMask,fgMask2, res,opening, detect_img;

        pBackSub = createBackgroundSubtractorMOG2();
        pBackSub2 =createBackgroundSubtractorKNN(5000,400, false);
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
            //std::string filename = filepath.filename();
            //imshow("1 window", fgMask);

            pBackSub->apply(dst, fgMask2);
            morphologyEx(fgMask2,opening, MORPH_ERODE, getStructuringElement(MORPH_ELLIPSE,Size{7,7}));
            morphologyEx(opening,opening, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE,Size{50,50}));

            //morphologyEx(dst,opening, MORPH_ERODE, getStructuringElement(MORPH_ELLIPSE,Size{17,17}));
            //morphologyEx(opening,opening, MORPH_DILATE, getStructuringElement(MORPH_ELLIPSE,Size{17,17}));

            //pBackSub->apply(opening, fgMask);
            imshow("origin window", opening);
            // Initialize HOG descriptor and use human detection classifier coefficients
            cv::HOGDescriptor hog;
            hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

            detect_img =opening;
            // Detect people and save them to detections
            std::vector<cv::Rect> detections;
            hog.detectMultiScale(detect_img, detections, 0, cv::Size(8, 8), cv::Size(32, 32), 1.2, 2);
            
            for (auto& detection : detections) {
                    //ResizeBoxes(detection);
                    cv::rectangle(detect_img, detection.tl(), detection.br(), cv::Scalar(255, 0, 0), 2);
                }
            
            imshow("2 window", detect_img);

            //imshow("result window", res);
            int k = waitKey(30); // Wait for a keystroke in the window
            
            //imwrite(std::regex_replace(image_path, pattern, "outputDataset"), dst);
        }

    

    }
    return 0;
}