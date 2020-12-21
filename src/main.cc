/*
 * main.cc
 *
 *  Created on: Dec 21, 2020
 *      Author: amyznikov
 */

#include "c_polar_stereo_rectification.h"
#include <opencv2/core/ocl.hpp>
#include <opencv2/xfeatures2d.hpp>


/**
 * @brief Detect and match SURF keypoints on pair of images
 * */
static bool detect_and_match_key_points(const cv::Mat images[2],
    std::vector<cv::Point2f> output_matched_points[2])
{

  /**
   * @brief David Lowe ratio test nearest / second nearest < ratio
   * */
  static const auto lowe_ratio_test =
      [](const std::vector<std::vector<cv::DMatch>> & matches12,
          std::vector<cv::DMatch> * output_good_matches,
          double lowe_ratio = 0.8)
      {
        output_good_matches->clear();
        output_good_matches->reserve(matches12.size()); /* optimistic guess all matches are good */

        for ( const std::vector<cv::DMatch> & m : matches12 ) {
          if ( m.size() == 1 || (m.size() == 2 && m[0].distance < lowe_ratio * m[1].distance) )
          output_good_matches->emplace_back(m[0]);
        }

      };


  /*
   * SURF detector parameters
   * */
  static const struct {
    double hessianThreshold = 200;
    int nOctaves = 2;
    int nOctaveLayers = 1;
    bool extended = false;
    bool upright = true;
  } SURF;

  /*
   * Create SURF detector
   * */
  cv::Ptr<cv::xfeatures2d::SURF> detector =
      cv::xfeatures2d::SURF::create(
          SURF.hessianThreshold,
          SURF.nOctaves,
          SURF.nOctaveLayers,
          SURF.extended,
          SURF.upright);


  /*
   * Detect SURF keypoints and compute their descriptors
   * */
  std::vector<cv::KeyPoint> keypoints[2];
  cv::Mat descriptors[2];

  for ( int i = 0; i < 2; ++i ) {

    detector->detectAndCompute(images[i], cv::noArray(),
        keypoints[i],
        descriptors[i]);

  }


  /*
   * USE FLANN based matcher to match detected keypoints
   * */
  cv::Ptr<cv::FlannBasedMatcher> keypoints_matcher;

  if ( descriptors[0].depth() == CV_32F ) {
    keypoints_matcher = cv::makePtr<cv::FlannBasedMatcher>(
        cv::makePtr<cv::flann::KDTreeIndexParams>(1),
        cv::makePtr<cv::flann::SearchParams>(cvflann::FLANN_CHECKS_UNLIMITED, 0, false));  // 32
  }
  else {
    keypoints_matcher = cv::makePtr<cv::FlannBasedMatcher>(
        cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1),
        cv::makePtr<cv::flann::SearchParams>(cvflann::FLANN_CHECKS_UNLIMITED, 0, false));
  }



  /*
   * Match keypoints
   * */
  std::vector<std::vector<cv::DMatch> > matches12;

  keypoints_matcher->knnMatch(descriptors[0], descriptors[1],
      matches12, 2);



  /*
   * Use David Lowe ratio test to filter best matches
   * */
  std::vector<cv::DMatch> best_matches;

  lowe_ratio_test(matches12,
      &best_matches);

  fprintf(stderr, "%s(): best_matches.size()=%zu \n",
      __func__,
      best_matches.size());



  /*
   * Return only best matches to caller
   * */
  for ( int i = 0; i < 2; ++i ) {
    output_matched_points[i].clear();
    output_matched_points[i].reserve(best_matches.size());
  }

  for ( int i = 0, n = best_matches.size(); i < n; ++i ) {
    output_matched_points[0].emplace_back(keypoints[0][best_matches[i].queryIdx].pt);
    output_matched_points[1].emplace_back(keypoints[1][best_matches[i].trainIdx].pt);
  }



  /*
   * Minimum 7 points is required for fundamentakl matrix estimation
   * */
  return best_matches.size() >= 7;
}


/**
 *  @brief Find fundamental matrix using matched keypoints
 * */
static bool find_fundamental_matrix(const cv::Mat images[2],
    cv::Matx33d & output_fundamental_matrix)
{

  /*
   * Detect and match keypoints on pair of input images
   * */
  std::vector<cv::Point2f> matched_keypoints[2];

  if ( !detect_and_match_key_points(images, matched_keypoints) ) {

    fprintf(stderr, "%s(): detect_and_match_key_points() fails\n",
        __func__);

    return false;
  }



  /*
   * Find fundamental matrix using matched points
   * */
  const cv::Mat F =
      cv::findFundamentalMat(matched_keypoints[0], matched_keypoints[1],
          cv::noArray(),
          cv::FM_LMEDS);

  if ( F.empty() ) {

    fprintf(stderr, "%s(): findFundamentalMat() fails\n",
        __func__);

    return false;
  }



  /*
   * Return fundamental maxtrix to caller
   * */
  output_fundamental_matrix = F;

  return true;
}


/**
 * @brief c_polar_stereo_rectification tester
 */
int main(int argc, char *argv[])
{
  /* Input images file names */
  std::string input_file_names[2];

  /* Input images*/
  cv::Mat input_images[2];

  /* Forward remapped (rectified) input images */
  cv::Mat forward_transfrormed_images[2];

  /* Reverse remapped rectified images */
  cv::Mat reverse_transfrormed_images[2];

  /* Fundamental_matrix */
  cv::Matx33d F;  //

  /* Stereo rectificator */
  c_polar_stereo_rectification rectification;

  /*
   * Parse command line arguments
   * */
  for ( int i = 1; i < argc; ++i ) {

    if ( strcmp(argv[i], "--help") == 0 ) {

      fprintf(stdout,
          "Usage:\n"
          "    PolarRectification [OPTIONS] image1 image2\n"
          "\n"
          "OPTIONS:\n"
          "   NONE\n"
          "\n"

      );
      return 0;
    }

    if ( input_file_names[0].empty() ) {  /* first input file name */
      input_file_names[0] = argv[i];
    }

    else if ( input_file_names[1].empty() ) {  /* second input file name */
      input_file_names[1] = argv[i];
    }

    else {  /* unparsed command line argument */
      fprintf(stderr, "Command line error: invalid argument %s\n",
          argv[i]);
      return 1;
    }
  }

  /*
   * Check if exactly 2 input file names are provided
   */
  for ( int i = 0; i < 2; ++i ) {
    if ( input_file_names[i].empty() ) {
      fprintf(stderr, "Two input images expected\n");
      return 1;
    }
  }



  /*
   * My OpenCV build has problems with OCL on my machine.
   * Comment out this line if you want to allow OCL in OpenCV
   */
  cv::ocl::setUseOpenCL(false);


  /*
   * Load input images
   */
  for ( int i = 0; i < 2; ++i ) {
    input_images[i] = cv::imread(input_file_names[i], cv::IMREAD_UNCHANGED);
    if ( !input_images[i].data ) {
      fprintf(stderr, "Can not read image '%s'\n",
          input_file_names[i].c_str());
      return 1;
    }
  }



  /*
   * Find fundamental matrix using pair of input images
   * */
  if ( !find_fundamental_matrix(input_images, F) ) {
    fprintf(stderr, "find_fundamental_matrix() fails\n");
    return 1;
  }

  fprintf(stderr, "F: {\n"
      "%+12.6f\t%+12.6f\t%+12.6f\n"
      "%+12.6f\t%+12.6f\t%+12.6f\n"
      "%+12.6f\t%+12.6f\t%+12.6f\n"
      "}\n",
      F(0,0),F(0,1),F(0,2),
      F(1,0),F(1,1),F(1,2),
      F(2,0),F(2,1),F(2,2));


  /*
   * Compute rectification mappings using fundamental maxtrix and image size provided
   * */
  if ( !rectification.compute(F, input_images[0].size()) ) {
    fprintf(stderr, "rectification.compute() fails\n");
    return 1;
  }


  /*
   * Remap (stereo rectify) input images
   * */
  rectification.remap(input_images[0], forward_transfrormed_images[0],
      input_images[1], forward_transfrormed_images[1]);


  /*
   * Reverse remap rectified images
   * */
  rectification.unmap(forward_transfrormed_images[0], reverse_transfrormed_images[0],
      forward_transfrormed_images[1], reverse_transfrormed_images[1]);

  /*
   * Dump input and transformaed images to disk for visual inspection
   * */

  cv::imwrite("input0.png",
      input_images[0]);

  cv::imwrite("input1.png",
      input_images[1]);

  cv::imwrite("forward0.png",
      forward_transfrormed_images[0]);

  cv::imwrite("forward1.png",
      forward_transfrormed_images[1]);

  cv::imwrite("reverse0.png",
      reverse_transfrormed_images[0]);

  cv::imwrite("reverse1.png",
      reverse_transfrormed_images[1]);


  /*
   * Finish this test
   */
  return 0;
}

