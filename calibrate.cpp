#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

int board_w = 9;
int board_h = 6;
float squareSize = 0.02423; //0.02423m

cv::Size board_sz = cv::Size(board_w, board_h);
int board_n = board_w * board_h;

vector<vector<cv::Point3d> > object_points;
vector<vector<cv::Point2f> > imagePoints1, imagePoints2;
vector<cv::Point2f> corners1, corners2;
vector< vector< Point2d > > left_img_points, right_img_points;

Mat img1, img2, gray1, gray2, spl1, spl2, Fimg1, Fimg2;

void load_image_points() {
  int num_imgs = 29;

  spl1 = imread("/Users/sourishghosh/vision/fisheye_stereo/imgs/left10.jpg", CV_LOAD_IMAGE_COLOR);
  spl2 = imread("/Users/sourishghosh/vision/fisheye_stereo/imgs/right10.jpg", CV_LOAD_IMAGE_COLOR);

  for (int i = 1; i <= num_imgs; i++) {
    char left_img[100], right_img[100];
    sprintf(left_img, "/Users/sourishghosh/vision/fisheye_stereo/imgs/left%d.jpg", i);
    sprintf(right_img, "/Users/sourishghosh/vision/fisheye_stereo/imgs/right%d.jpg", i);
    img1 = imread(left_img, CV_LOAD_IMAGE_COLOR);
    img2 = imread(right_img, CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(img1, gray1, CV_BGR2GRAY);
    cv::cvtColor(img2, gray2, CV_BGR2GRAY);

    bool found1 = false, found2 = false;

    found1 = cv::findChessboardCorners(img1, board_sz, corners1,
  CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
    found2 = cv::findChessboardCorners(img2, board_sz, corners2,
  CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

    if (found1)
    {
      cv::cornerSubPix(gray1, corners1, cv::Size(5, 5), cv::Size(-1, -1),
  cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
      cv::drawChessboardCorners(gray1, board_sz, corners1, found1);
    }
    if (found2)
    {
      cv::cornerSubPix(gray2, corners2, cv::Size(5, 5), cv::Size(-1, -1),
  cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
      cv::drawChessboardCorners(gray2, board_sz, corners2, found2);
    }

    vector<cv::Point3d> obj;
    for( int i = 0; i < board_h; ++i )
      for( int j = 0; j < board_w; ++j )
        obj.push_back(Point3d(double( (float)j * squareSize ), double( (float)i * squareSize ), 0));

    if (found1 && found2) {
      cout << i << ". Found corners!" << endl;
      imagePoints1.push_back(corners1);
      imagePoints2.push_back(corners2);
      object_points.push_back(obj);
    }
  }
  for (int i = 0; i < imagePoints1.size(); i++) {
    vector< Point2d > v1, v2;
    for (int j = 0; j < imagePoints1[i].size(); j++) {
      v1.push_back(Point2d((double)imagePoints1[i][j].x, (double)imagePoints1[i][j].y));
      v2.push_back(Point2d((double)imagePoints2[i][j].x, (double)imagePoints2[i][j].y));
    }
    left_img_points.push_back(v1);
    right_img_points.push_back(v2);
  }
}

int main(int argc, char const *argv[])
{
  load_image_points();

  printf("Starting Calibration\n");
  cv::Matx33d K1, K2, R;
  cv::Vec3d T;
  cv::Vec4d D1, D2;
  int flag = 0;
  flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
  flag |= cv::fisheye::CALIB_CHECK_COND;
  flag |= cv::fisheye::CALIB_FIX_SKEW;
  //flag |= cv::fisheye::CALIB_FIX_K2;
  //flag |= cv::fisheye::CALIB_FIX_K3;
  //flag |= cv::fisheye::CALIB_FIX_K4;
  cv::fisheye::stereoCalibrate(object_points, left_img_points, right_img_points,
      K1, D1, K2, D2, img1.size(), R, T, flag,
      cv::TermCriteria(3, 12, 0));

  cv::FileStorage fs1("mystereocalib.yml", cv::FileStorage::WRITE);
  fs1 << "K1" << Mat(K1);
  fs1 << "K2" << Mat(K2);
  fs1 << "D1" << D1;
  fs1 << "D2" << D2;
  fs1 << "R" << Mat(R);
  fs1 << "T" << T;
  printf("Done Calibration\n");

  printf("Starting Rectification\n");

  cv::Mat R1, R2, P1, P2, Q;
  cv::fisheye::stereoRectify(K1, D1, K2, D2, img1.size(), R, T, R1, R2, P1, P2, 
Q, CV_CALIB_ZERO_DISPARITY, img1.size(), 0.0, 1.1);

  fs1 << "R1" << R1;
  fs1 << "R2" << R2;
  fs1 << "P1" << P1;
  fs1 << "P2" << P2;
  fs1 << "Q" << Q;

  printf("Done Rectification\n");

  cv::Mat lmapx, lmapy, rmapx, rmapy;
  Mat imgU1, imgU2;
  cv::fisheye::initUndistortRectifyMap(K1, D1, R1, P1, spl1.size(), CV_32F, lmapx, lmapy);
  cv::fisheye::initUndistortRectifyMap(K2, D2, R2, P2, spl1.size(), CV_32F, rmapx, rmapy);

  cv::remap(spl1, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
  cv::remap(spl2, imgU2, rmapx, rmapy, cv::INTER_LINEAR);

  imshow("image1", imgU1);
  imshow("image2", imgU2);

  waitKey(0);

  return 0;
}
