  
#include "CameraCalibrator.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <cstdlib>
#include <random>
#include <vector>

#include "opencv2/features2d/features2d.hpp"

#include "opencv2/xfeatures2d.hpp"

#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>



using namespace std;
using namespace cv;



int main(){
	cv::Mat img1, img2;

	img1 = cv::imread("imR.png",cv::IMREAD_GRAYSCALE);
	img2 = cv::imread("imL.png",cv::IMREAD_GRAYSCALE);

	// Define keypoints vector
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;
	// Define feature detector
	cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SIFT::create(74);	
	// Keypoint detection
	ptrFeature2D->detect(img1,keypoints1);
	ptrFeature2D->detect(img2,keypoints2);
	// Extract the descriptor
	cv::Mat descriptors1;
	cv::Mat descriptors2;

	ptrFeature2D->compute(img1,keypoints1,descriptors1);
	ptrFeature2D->compute(img2,keypoints2,descriptors2);
	// Construction of the matcher
	cv::BFMatcher matcher(cv::NORM_L2);
	// Match the two image descriptors
	std::vector<cv::DMatch> outputMatches;
	matcher.match(descriptors1,descriptors2, outputMatches);

	// Convert keypoints into Point2f
	std::vector<cv::Point2f> points1, points2;
	for (std::vector<cv::DMatch>::const_iterator it= outputMatches.begin();
			 it!= outputMatches.end(); ++it) {

			 // Get the position of left keypoints
			 points1.push_back(keypoints1[it->queryIdx].pt);
			 // Get the position of right keypoints
			 points2.push_back(keypoints2[it->trainIdx].pt);
	    }


	// Compute F matrix from 7 matches
	// cv::Mat fundamental= cv::findFundamentalMat(selPoints1, selPoints2, cv::FM_7POINT);
	std::vector<uchar> inliers(points1.size(),0);
	cv::Mat fundamental= cv::findFundamentalMat(
		points1,points2, // matching points
	    inliers,         // match status (inlier or outlier)  
	    cv::FM_RANSAC,   // RANSAC method
	    1.0,        // distance to epipolar line
	    0.98);     // confidence probability
	

	// std::vector<cv::DMatch> matches;
	// cv::Mat fundamental = ransacTest(outputMatches, keypoints1, keypoints2, matches);

	cout<<fundamental;


	// Compute homographic rectification
	cv::Mat h1, h2;
	cv::stereoRectifyUncalibrated(points1, points2,
	fundamental,
	img1.size(), h1, h2);
	// Rectify the images through warping
	cv::Mat rectified1;
	cv::warpPerspective(img1, rectified1, h1, img1.size());
	cv::Mat rectified2;
	cv::warpPerspective(img2, rectified2, h2, img1.size());


	// Compute disparity
	cv::Mat disparity;
	cv::Ptr<cv::StereoMatcher> pStereo = cv::StereoSGBM::create(0, 32, 5);
	pStereo->compute(rectified1, rectified2, disparity);

	cv::imwrite("disparity.jpg", disparity);







}

