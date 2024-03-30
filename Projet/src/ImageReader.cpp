#include <omp.h>

#include <filesystem>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <regex>
#include <set>
#include <string>
#include <utility>
using namespace cv;
namespace fs = std::filesystem;

// This program is used to denormalize all the images

std::vector<fs::path> retrieveVideoImagePaths() {
    std::string path = "../../dataset/";
    std::vector<std::string> dataset_path;
    std::set<fs::path> sorted_by_name;
    for (const auto &entry : std::filesystem::directory_iterator{path}) {
        std::string dirname = entry.path().filename();

        fs::create_directories("../../outputDataset/" + dirname);
        for (const auto &file_entry : std::filesystem::directory_iterator{entry.path()}) {
            sorted_by_name.emplace(file_entry.path());
        }
    }
    std::vector paths(sorted_by_name.begin(), sorted_by_name.end());
    return paths;
}

void applyMorphTransform(Mat &fgMask, Mat &element) {
    morphologyEx(fgMask, fgMask, MORPH_ERODE, element, Point(-1, -1), 5);
    morphologyEx(fgMask, fgMask, MORPH_DILATE, element, Point(-1, -1), 8);
}

void spotPeople(Mat &fgMask, double &total_time) {
    double start_time = omp_get_wtime();

    Mat labels, stats, centroids;
    int nbcomp = connectedComponentsWithStats(fgMask, labels, stats, centroids, 4, CV_32S);
    Point pt1, pt2;
    int area;
    for (int i = 1; i < nbcomp; i++) {  // Start at 1 because 0 is the background.
        area = stats.at<int>(i, CC_STAT_AREA);
        if (area > 2000) {
            pt1 = Point(stats.at<int>(i, CC_STAT_LEFT), stats.at<int>(i, CC_STAT_TOP));
            pt2 = Point(pt1.x + stats.at<int>(i, CC_STAT_WIDTH), pt1.y + stats.at<int>(i, CC_STAT_HEIGHT));
            // Scalar color = Scalar(255, 255, 0);
            rectangle(fgMask, pt1, pt2, 255);
        }
    }
    double end_time = omp_get_wtime();
    total_time += (end_time - start_time);
}

int main(int argc, char *argv[]) {
    int nbThreads = 1;
    if (argc >= 2) {
        nbThreads = std::stoi(argv[1]);
    }
    std::cout << "Nb threads : " << nbThreads << std::endl;

    double start_time_all = omp_get_wtime();
    double start_time, end_time;
    double total_time_spotPeople = 0, total_time_morphTransform = 0, total_time_equalize = 0, total_time_applyBackgroundRemover = 0;
    double total_time_readImg = 0, total_time_writeImg = 0;
    omp_set_num_threads(nbThreads);

    // ------- Retrieve all images -------
    start_time = omp_get_wtime();
    std::vector<fs::path> sorted_by_name = retrieveVideoImagePaths();
    end_time = omp_get_wtime();
    std::cout << "Total time taken in retrieveVideoImagePaths() : " << end_time - start_time << "seconds " << std::endl;

    std::regex pattern("dataset");
    fs::path current_parent_path = "./";

    std::vector<Mat> imgs(sorted_by_name.size());
    std::map<int, bool> valid_imgs;
    fs::path filepath;
    std::string image_path;

    start_time = omp_get_wtime();
#pragma omp parallel for private(image_path)
    for (size_t i = 0; i < sorted_by_name.size(); i++) {
        // Checking if it's still the same scene, if not, create a new BGSubtractor
        // ------- Read Image -------
        image_path = samples::findFile(sorted_by_name[i]);
        imgs[i] = imread(image_path, IMREAD_GRAYSCALE);
        if (imgs[i].empty()) {
            std::cout << "Could not read the image: " << image_path << std::endl;
            valid_imgs[i] = false;
            // return 1;
        } else {
            valid_imgs[i] = true;
        }
    }
    end_time = omp_get_wtime();
    total_time_readImg += (end_time - start_time);
    std::cout << "Total time taken for reading image : " << total_time_readImg << "seconds " << std::endl;


    // ------- Equalize Image -------
    start_time = omp_get_wtime();
#pragma omp parallel for private(start_time, end_time)
    for (size_t i = 0; i < sorted_by_name.size(); i++) {
        if (valid_imgs[i]) {
            equalizeHist(imgs[i], imgs[i]);
        }
    }
    end_time = omp_get_wtime();
    total_time_equalize += (end_time - start_time);
    std::cout << "Total time taken for equalizeHist() : " << total_time_equalize << "seconds " << std::endl;

    // ------- Apply background remover -------

    Ptr<BackgroundSubtractor> pBackSub;
    pBackSub = createBackgroundSubtractorMOG2();
    start_time = omp_get_wtime();

    for (size_t i = 0; i < sorted_by_name.size(); i++) {
        if (valid_imgs[i]) {
            const auto filepath = sorted_by_name[i];

            // Checking if it's still the same scene, if not, create a new BGSubtractor
            fs::path parent_path = filepath.parent_path();
            if (current_parent_path.compare(parent_path) != 0) {
                // std::cout << filepath.parent_path() << std::endl;
                current_parent_path = parent_path;
                pBackSub = createBackgroundSubtractorMOG2();
            }
            pBackSub->apply(imgs[i], imgs[i]);
        }
    }
    end_time = omp_get_wtime();
    total_time_applyBackgroundRemover += (end_time - start_time); 

    std::cout << "Total time taken for background Remover : " << total_time_applyBackgroundRemover << "seconds " << std::endl;
     
    // ------- Removing noise with morphological transformation -------
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    start_time = omp_get_wtime();
#pragma omp parallel for
    for (size_t i = 0; i < sorted_by_name.size(); i++) {
        if (valid_imgs[i]) {

            applyMorphTransform(imgs[i], element);
        }
    }

    end_time = omp_get_wtime();
    total_time_morphTransform = (end_time - start_time);
    std::cout << "Total time taken in  applyMorphTransform() : " << total_time_morphTransform << "seconds " << std::endl;

    // ------- Looking for ppl among connected components -------    
    start_time = omp_get_wtime();
#pragma omp parallel for
    for (size_t i = 0; i < sorted_by_name.size(); i++) {
        if (valid_imgs[i]) {
            spotPeople(imgs[i], total_time_spotPeople);
        }
    }
    end_time = omp_get_wtime();
    total_time_spotPeople = (end_time - start_time);
    std::cout << "Total time taken in  spotPeople() : " << total_time_spotPeople << "seconds " << std::endl;

    // ------- Read and Save Image ------- */
    start_time = omp_get_wtime();
#pragma omp parallel for
    for (size_t i = 0; i < sorted_by_name.size(); i++) {
        if (valid_imgs[i]) {
            // std::string filename = sorted_by_name[i].filename();
            // imshow("Display window", fgMask);
            // int k = waitKey(0); // Wait for a keystroke in the window

            imwrite(std::regex_replace(samples::findFile(sorted_by_name[i]), pattern, "outputDataset"), imgs[i]);

        }
    }            
    end_time = omp_get_wtime();
    total_time_writeImg += (end_time - start_time);
    std::cout << "Total time taken for writing image : " << total_time_writeImg << "seconds " << std::endl;

    double end_time_all = omp_get_wtime();

    std::cout << "========================== \n Total time taken overall : " << end_time_all - start_time_all << "seconds " << std::endl;

    return 0;
}