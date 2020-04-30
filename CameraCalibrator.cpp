/*------------------------------------------------------------------------------------------*\
   This file contains material supporting chapter 9 of the cookbook:  
   Computer Vision Programming using the OpenCV Library. 
   by Robert Laganiere, Packt Publishing, 2011.

   This program is free software; permission is hereby granted to use, copy, modify, 
   and distribute this source code, or portions thereof, for any purpose, without fee, 
   subject to the restriction that the copyright notice may not be removed 
   or altered from any source or altered source distribution. 
   The software is released on an as-is basis and without any warranties of any kind. 
   In particular, the software is not guaranteed to be fault-tolerant or free from failure. 
   The author disclaims all warranties with regard to this software, any use, 
   and any consequent failure, is purely the responsibility of the user.
 
   Copyright (C) 2010-2011 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

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

std::vector<cv::Mat> rvecs, tvecs;

// Open chessboard images and extract corner points
int CameraCalibrator::addChessboardPoints(
         const std::vector<std::string>& filelist, 
         cv::Size & boardSize) {

	// the points on the chessboard
  std::vector<cv::Point2f> imageCorners;
  std::vector<cv::Point3f> objectCorners;
    

    // 3D Scene Points:
    // Initialize the chessboard corners 
    // in the chessboard reference frame
	// The corners are at 3D location (X,Y,Z)= (i,j,0)
	for (int i=0; i<boardSize.height; i++) {
		for (int j=0; j<boardSize.width; j++) {

			objectCorners.push_back(cv::Point3f(i, j, 0.0f));
		}
    }

    // 2D Image points:
    cv::Mat image; // to contain chessboard image
    int successes = 0;
    // for all viewpoints
    for (int i=0; i<filelist.size(); i++) {

        // Open the image
        image = cv::imread(filelist[i],0);

        // Get the chessboard corners
        bool found = cv::findChessboardCorners(
                        image, boardSize, imageCorners);

        // Get subpixel accuracy on the corners
        cv::cornerSubPix(image, imageCorners, 
                  cv::Size(5,5), 
                  cv::Size(-1,-1), 
			cv::TermCriteria(cv::TermCriteria::MAX_ITER +
                          cv::TermCriteria::EPS, 
             30,		// max number of iterations 
             0.1));     // min accuracy

          // If we have a good board, add it to our data
		  if (imageCorners.size() == boardSize.area()) {

			// Add image and scene points from one view
            addPoints(imageCorners, objectCorners);
            successes++;
          }



        //Draw the corners
        cv::drawChessboardCorners(image, boardSize, imageCorners, found);
        cv::imshow("Corners on Chessboard", image);
        cv::waitKey(100);
    }

	return successes;
}

// Add scene points and corresponding image points
void CameraCalibrator::addPoints(const std::vector<cv::Point2f>& imageCorners, const std::vector<cv::Point3f>& objectCorners) {

	// 2D image points from one view
	imagePoints.push_back(imageCorners);          
	// corresponding 3D scene points
	objectPoints.push_back(objectCorners);

}

// Calibrate the camera
// returns the re-projection error
double CameraCalibrator::calibrate(cv::Size &imageSize)
{
	// undistorter must be reinitialized
	mustInitUndistort= true;

	//Output rotations and translations
    

	// start calibration
	return 
     calibrateCamera(objectPoints, // the 3D points
		      imagePoints,  // the image points
					imageSize,    // image size
					cameraMatrix, // output camera matrix
					distCoeffs,   // output distortion matrix
					rvecs, tvecs, // Rs, Ts 
					flag);        // set options
//					,CV_CALIB_USE_INTRINSIC_GUESS);

}

// remove distortion in an image (after calibration)
cv::Mat CameraCalibrator::remap(const cv::Mat &image) {

	cv::Mat undistorted;

	if (mustInitUndistort) { // called once per calibration
    
		cv::initUndistortRectifyMap(
			cameraMatrix,  // computed camera matrix
            distCoeffs,    // computed distortion matrix
            cv::Mat(),     // optional rectification (none) 
			cv::Mat(),     // camera matrix to generate undistorted
			cv::Size(640,480),
//            image.size(),  // size of undistorted
            CV_32FC1,      // type of output map
            map1, map2);   // the x and y mapping functions

		mustInitUndistort= false;
	}

	// Apply mapping functions
    cv::remap(image, undistorted, map1, map2, 
		cv::INTER_LINEAR); // interpolation type

	return undistorted;
}


// Set the calibration options
// 8radialCoeffEnabled should be true if 8 radial coefficients are required (5 is default)
// tangentialParamEnabled should be true if tangeantial distortion is present
void CameraCalibrator::setCalibrationFlag(bool radial8CoeffEnabled, bool tangentialParamEnabled) {

    // Set the flag used in cv::calibrateCamera()
    flag = 0;
    if (!tangentialParamEnabled) flag += CV_CALIB_ZERO_TANGENT_DIST;
	if (radial8CoeffEnabled) flag += CV_CALIB_RATIONAL_MODEL;
}


cv::Vec3d CameraCalibrator::triangulate(const cv::Mat &p1, const cv::Mat &p2, const cv::Vec2d &u1, const cv::Vec2d &u2) {

  // system of equations assuming image=[u,v] and X=[x,y,z,1]
  // from u(p3.X)= p1.X and v(p3.X)=p2.X
  cv::Matx43d A(u1(0)*p1.at<double>(2, 0) - p1.at<double>(0, 0),
  u1(0)*p1.at<double>(2, 1) - p1.at<double>(0, 1),
  u1(0)*p1.at<double>(2, 2) - p1.at<double>(0, 2),
  u1(1)*p1.at<double>(2, 0) - p1.at<double>(1, 0),
  u1(1)*p1.at<double>(2, 1) - p1.at<double>(1, 1),
  u1(1)*p1.at<double>(2, 2) - p1.at<double>(1, 2),
  u2(0)*p2.at<double>(2, 0) - p2.at<double>(0, 0),
  u2(0)*p2.at<double>(2, 1) - p2.at<double>(0, 1),
  u2(0)*p2.at<double>(2, 2) - p2.at<double>(0, 2),
  u2(1)*p2.at<double>(2, 0) - p2.at<double>(1, 0),
  u2(1)*p2.at<double>(2, 1) - p2.at<double>(1, 1),
  u2(1)*p2.at<double>(2, 2) - p2.at<double>(1, 2));

  cv::Matx41d B(p1.at<double>(0, 3) - u1(0)*p1.at<double>(2,3),
                p1.at<double>(1, 3) - u1(1)*p1.at<double>(2,3),
                p2.at<double>(0, 3) - u2(0)*p2.at<double>(2,3),
                p2.at<double>(1, 3) - u2(1)*p2.at<double>(2,3));

  // X contains the 3D coordinate of the reconstructed point
  cv::Vec3d X;
  // solve AX=B
  cv::solve(A, B, X, cv::DECOMP_SVD);
  return X;
}

// triangulate a vector of image points
void CameraCalibrator::triangulate(const cv::Mat &p1, const cv::Mat &p2, const std::vector<cv::Vec2d> &pts1, const std::vector<cv::Vec2d> &pts2, std::vector<cv::Vec3d> &pts3D) {

  for (int i = 0; i < pts1.size(); i++) {

    pts3D.push_back(triangulate(p1, p2, pts1[i], pts2[i]));
  }
}




int main(){




  cout<<"compiled"<<endl;

  const std::vector<std::string> files = {"chess1.jpg", "chess2.jpg"};
  cv::Size board_size(7,7);

  CameraCalibrator cal;
  cal.addChessboardPoints(files, board_size);

  cv::Mat img = cv::imread("chess1.jpg");

  cv::Size img_size = img.size();
  cal.calibrate(img_size);
  cout<<cameraMatrix<<endl;


  cv::Mat image1 = cv::imread("m3.jpg");
  cv::Mat image2 = cv::imread("m4.jpg");

  // vector of keypoints and descriptors
  std::vector<cv::KeyPoint> keypoints1;
  std::vector<cv::KeyPoint> keypoints2;
  cv::Mat descriptors1, descriptors2;

  // Construction of the SIFT feature detector
  cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SIFT::create(500);

  // Detection of the SIFT features and associated descriptors
  ptrFeature2D->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
  ptrFeature2D->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

  // Match the two image descriptors
  // Construction of the matcher with crosscheck
  cv::BFMatcher matcher(cv::NORM_L2, true);
  std::vector<cv::DMatch> matches;
  matcher.match(descriptors1, descriptors2, matches);

  // Convert keypoints into Point2f
  std::vector<cv::Point2f> points1, points2;

  for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
    // Get the position of left keypoints
    float x = keypoints1[it->queryIdx].pt.x;
    float y = keypoints1[it->queryIdx].pt.y;
    points1.push_back(cv::Point2f(x, y));
    // Get the position of right keypoints
    x = keypoints2[it->trainIdx].pt.x;
    y = keypoints2[it->trainIdx].pt.y;
    points2.push_back(cv::Point2f(x, y));
  }

  // Find the essential between image 1 and image 2
  cv::Mat inliers;
  cv::Mat essential = cv::findEssentialMat(points1, points2, cameraMatrix, cv::RANSAC, 0.9, 1.0, inliers);

  cout<<essential<<endl;

  // recover relative camera pose from essential matrix
  cv::Mat rotation, translation;
  cv::recoverPose(essential, points1, points2, cameraMatrix, rotation, translation, inliers);
  cout<<rotation<<endl;
  cout<<translation<<endl;


  // compose projection matrix from R,T
  cv::Mat projection2(3, 4, CV_64F); // the 3x4 projection matrix
  rotation.copyTo(projection2(cv::Rect(0, 0, 3, 3)));
  translation.copyTo(projection2.colRange(3, 4));
  // compose generic projection matrix
  cv::Mat projection1(3, 4, CV_64F, 0.); // the 3x4 projection matrix
  cv::Mat diag(cv::Mat::eye(3, 3, CV_64F));
  diag.copyTo(projection1(cv::Rect(0, 0, 3, 3)));
  // to contain the inliers
  std::vector<cv::Vec2d> inlierPts1;
  std::vector<cv::Vec2d> inlierPts2;
  // create inliers input point vector for triangulation
  int j(0);
  for (int i = 0; i < inliers.rows; i++) {
    if (inliers.at<uchar>(i)) {
      inlierPts1.push_back(cv::Vec2d(points1[i].x, points1[i].y));
      inlierPts2.push_back(cv::Vec2d(points2[i].x, points2[i].y));
    }
  }
  // undistort and normalize the image points
  std::vector<cv::Vec2d> points1u;
  cv::undistortPoints(inlierPts1, points1u, cameraMatrix, distCoeffs);
  std::vector<cv::Vec2d> points2u;
  cv::undistortPoints(inlierPts2, points2u, cameraMatrix, distCoeffs);
  

  // Triangulation
  std::vector<cv::Vec3d> points3D;
  cal.triangulate(projection1, projection2, points1u, points2u, points3D);

  cout<<"3D points :"<<points3D.size()<<endl;

  viz::Viz3d window; //creating a Viz window

  // viz::WCameraPosition cam(cameraMatrix, image1, 30.0);


 

  



  //Displaying the Coordinate Origin (0,0,0)
  window.showWidget("coordinate", viz::WCoordinateSystem());
  //Displaying the 3D points in green
  // cv::viz::WCameraPosition camm(cameraMatrix, image1, 30.0, viz::Color::black());

  // window.showWidget("Camera", camm);
  
  window.setBackgroundColor(cv::viz::Color::black());


  // cv::Mat rvec, tvec;
  // cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
  // cv::Mat rot;
  // cv::Rodrigues(rvec, rot);


  // cv::Affine3d pose(rot, tvec);
  // window.setWidgetPose("top", pose);



  window.showWidget("points", viz::WCloud(points3D, viz::Color::green()));
  window.spin();





}