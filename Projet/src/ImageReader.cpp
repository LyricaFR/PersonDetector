#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <filesystem>
#include <string>
#include <set>
#include <omp.h>
#include <regex>
using namespace cv;
namespace fs = std::filesystem;

// This program is used to denormalize all the images

std::set<fs::path> retrieveVideoImagePaths(){
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
    return sorted_by_name;
}


void applyMorphTransform(Mat &fgMask,double &total_time_morphTransform){
    static Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

    double start_time = omp_get_wtime();
    morphologyEx(fgMask, fgMask, MORPH_ERODE, element, Point(-1, -1), 5); 
    morphologyEx(fgMask, fgMask, MORPH_DILATE, element, Point(-1, -1), 8); 
    double end_time = omp_get_wtime();
    total_time_morphTransform += (end_time - start_time);

}

void spotPeople(Mat &fgMask, double &total_time){
    double start_time = omp_get_wtime();

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
    double end_time = omp_get_wtime();
    total_time += (end_time - start_time);
}

int main()
{

    double start_time_all = omp_get_wtime();
    double start_time, end_time;
    double total_time_spotPeople = 0, total_time_morphTransform = 0,total_time_equalize = 0, total_time_applyBackgroundRemover = 0;
    double total_time_readImg = 0;


    Ptr<BackgroundSubtractor> pBackSub;
    pBackSub = createBackgroundSubtractorMOG2();

    // ------- Retrieve all images ------- 
    start_time = omp_get_wtime();
    std::set<fs::path> sorted_by_name = retrieveVideoImagePaths();
    end_time = omp_get_wtime();
    std::cout << "Total time taken in retrieveVideoImagePaths() : "<<end_time - start_time << "seconds "<< std::endl;

    std::regex pattern("dataset");
    Mat fgMask;
    fs::path current_parent_path = "./";
    
    for (const auto filepath : sorted_by_name){

        // Checking if it's still the same scene, if not, create a new BGSubtractor
        fs::path parent_path = filepath.parent_path();
        if (current_parent_path.compare(parent_path) != 0){
            std::cout << filepath.parent_path() << std::endl;
            current_parent_path = parent_path;
            pBackSub = createBackgroundSubtractorMOG2();
        }

        // ------- Read Image ------- 

        start_time = omp_get_wtime();
        std::string image_path = samples::findFile(filepath);
        Mat img = imread(image_path, IMREAD_GRAYSCALE);
        if(img.empty())
        {
            std::cout << "Could not read the image: " << image_path << std::endl;
            return 1;
        }

        end_time = omp_get_wtime();
        total_time_readImg += (end_time - start_time);

        Mat dst;

        // ------- Equalize Image ------- 
        start_time = omp_get_wtime();
        equalizeHist(img, dst);
        end_time = omp_get_wtime();
        total_time_equalize += (end_time - start_time);

        // ------- Apply background remover ------- 
        start_time = omp_get_wtime();
        pBackSub->apply(dst, fgMask);
        end_time = omp_get_wtime();
        total_time_applyBackgroundRemover += (end_time - start_time);

        // ------- Removing noise with morphological transformation -------

        applyMorphTransform(fgMask,total_time_morphTransform);

        // ------- Looking for ppl among connected components -------

        spotPeople(fgMask,total_time_spotPeople);

        // ----------------------------------------------------------

        std::string filename = filepath.filename();
        imshow("Display window", fgMask);
        //int k = waitKey(0.005); // Wait for a keystroke in the window
        
        //imwrite(std::regex_replace(image_path, pattern, "outputDataset"), dst);

    }
    double end_time_all = omp_get_wtime();

    std::cout << "Total time taken for reading Image : "<<total_time_readImg << "seconds "<< std::endl;
    std::cout << "Total time taken for equalizeHist() : "<<total_time_equalize << "seconds "<< std::endl;
    std::cout << "Total time taken for background Remover : "<<total_time_applyBackgroundRemover << "seconds "<< std::endl;
    std::cout << "Total time taken in  applyMorphTransform() : "<<total_time_morphTransform << "seconds "<< std::endl;
    std::cout << "Total time taken in  spotPeople() : "<<total_time_spotPeople << "seconds "<< std::endl;
    std::cout << "========================== \n Total time taken overall : "<<end_time_all - start_time_all << "seconds "<< std::endl;

    return 0;
}