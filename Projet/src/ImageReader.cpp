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
using namespace cv;
namespace fs = std::filesystem;

// This program is used to denormalize all the images

int main()
{

    Ptr<BackgroundSubtractor> pBackSub;
    pBackSub = createBackgroundSubtractorMOG2();

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
    Mat fgMask;
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    fs::path current_parent_path = "./";
    
    for (const auto filepath : sorted_by_name){

        // Checking if it's still the same scene, if not, create a new BGSubtractor
        fs::path parent_path = filepath.parent_path();
        if (current_parent_path.compare(parent_path) != 0){
            std::cout << filepath.parent_path() << std::endl;
            current_parent_path = parent_path;
            pBackSub = createBackgroundSubtractorMOG2();
        }


        std::string image_path = samples::findFile(filepath);
        Mat img = imread(image_path, IMREAD_GRAYSCALE);
        if(img.empty())
        {
            std::cout << "Could not read the image: " << image_path << std::endl;
            return 1;
        }
        Mat dst;
        equalizeHist(img, dst);
        pBackSub->apply(dst, fgMask);

        // ------- Removing noise with morphological transformation -------

        morphologyEx(fgMask, fgMask, MORPH_ERODE, element, Point(-1, -1), 5); 
        morphologyEx(fgMask, fgMask, MORPH_DILATE, element, Point(-1, -1), 8); 

        // ------- Looking for ppl among connected components -------

        Mat labels, stats, centroids;
        int nbcomp = connectedComponentsWithStats(fgMask, labels, stats, centroids, 4, CV_32S);
        for (int i = 1; i < nbcomp; i++){  // Start at 1 because 0 is the background.
            int area = stats.at<int>(i, CC_STAT_AREA);
            if (area > 2000){
                Point pt1 = Point(stats.at<int>(i, CC_STAT_LEFT), stats.at<int>(i, CC_STAT_TOP));
                Point pt2 = Point(pt1.x + stats.at<int>(i, CC_STAT_WIDTH), pt1.y + stats.at<int>(i, CC_STAT_HEIGHT));
                // Scalar color = Scalar(255, 255, 0);
                rectangle(fgMask, pt1, pt2, 255);
            }
        }

        // ----------------------------------------------------------

        std::string filename = filepath.filename();
        imshow("Display window", fgMask);
        int k = waitKey(0); // Wait for a keystroke in the window
        
        imwrite(std::regex_replace(image_path, pattern, "outputDataset"), dst);

    }
    return 0;
}