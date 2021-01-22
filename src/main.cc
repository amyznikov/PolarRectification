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
 * @brief Dump 3x3 matrix
 * */
static void pmat(const cv::Matx33d & F, const std::string & name)
{
  fprintf(stderr, "%s: {\n"
      "%+12.6f\t%+12.6f\t%+12.6f\n"
      "%+12.6f\t%+12.6f\t%+12.6f\n"
      "%+12.6f\t%+12.6f\t%+12.6f\n"
      "}\n",
      name.c_str(),
      F(0, 0), F(0, 1), F(0, 2),
      F(1, 0), F(1, 1), F(1, 2),
      F(2, 0), F(2, 1), F(2, 2));

}

/**
 * @brief Dump 3x1 vector
 * */
static void pmat(const cv::Vec3d & T, const std::string & name)
{
  fprintf(stderr, "%s: {\n"
      "%+12.6f\n"
      "%+12.6f\n"
      "%+12.6f\n"
      "}\n",
      name.c_str(),
      T(0),
      T(1),
      T(2));

}


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
    double hessianThreshold = 100;
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
   * Minimum 7 points is required for fundamental matrix estimation
   * */
  return best_matches.size() >= 7;
}

/**
 * @brief Compute median value and direction of the movements of keypoints beyween frames.
 *
 * The key points are expected to move between frames along epipolar lines from or toward to epipole.
 * This routine computes the median offset and it's direction (sign).
 * */
static double median_radial_flow(const cv::Point2d & e0, const cv::Point2d & e1,
    const std::vector<cv::Point2f> & keypoints0, const std::vector<cv::Point2f> & keypoints1,
    const cv::Mat1b & mask)
{
  std::vector<double> offsets;

  offsets.reserve(mask.rows);

  /*
   * Assume that optical axes of both frames are already parallel,
   * therefore epipoles must coinside on both frames (pixel coordinates must be equal
   * */
  const cv::Point2d e = 0.5 * (e0 + e1);

  for ( int i = 0; i < mask.rows; ++i ) {
    if ( mask[i] ) {

      const cv::Point2f & kp0 = keypoints0[i];
      const cv::Point2f & kp1 = keypoints1[i];

      const double r0 = cv::norm(cv::Vec2d(kp0.x - e.x, kp0.y - e.y));
      const double r1 = cv::norm(cv::Vec2d(kp1.x - e.x, kp1.y - e.y));

      offsets.emplace_back(r1 - r0);
    }
  }

  std::sort(offsets.begin(), offsets.end());

  return offsets[offsets.size() / 2];
}


/**
 *  @brief Combine findEssentialMat() and recoverPose() into single call
 *  and estimate rotation matrix and translation direction betweeen two frames
 *  given by matched key points.
 * */

static bool estimate_pose(const cv::Mat input_images[2],
    const cv::Matx33d & cameraMatrix,
    cv::Matx33d & output_rotation_matrix,
    cv::Vec3d & output_translation_vector,
    cv::Matx33d & output_fundamental_matrix,
    cv::Matx33d & output_homography1,
    double * output_median_radial_flow)
{
  cv::Matx33d R, H;
  cv::Matx33d F, E;
  cv::Vec3d T;
  cv::Mat1b mask;
  cv::Point2d epipoles[2];
  int ngood;


  /*
   * Detect and matcht key points on input images
   * */

  std::vector<cv::Point2f> matched_keypoints[2];

  if ( !detect_and_match_key_points(input_images, matched_keypoints) ) {
    fprintf(stderr, "%s(): %d detect_and_match_key_points() fails\n", __func__, __LINE__);
    return false;
  }

  /*
   * Initial estimation of fundaemntal matrix using pairs of matched keypoint
   */
  F = cv::findFundamentalMat(matched_keypoints[0], matched_keypoints[1],
      mask,
      cv::FM_LMEDS,
      0.5,
      0.999);

  /* Dump some debug info */
  fprintf(stderr, "findFundamentalMat(): good points = %d / %d\n", cv::countNonZero(mask), mask.rows);
  pmat(F, "F1");



  /*
   * Compute Essential Matrix from Fundamental Matrix
   * */
  E = cameraMatrix.t() * F * cameraMatrix;
  pmat(E, "E");

  /*
   * Use of cv::recoverPose() for estimation of camera rotation between frames
   * */
  ngood = cv::recoverPose(E,
      matched_keypoints[0],
      matched_keypoints[1],
      cameraMatrix,
      R, T,
      mask);

  /* Dump some debug info */
  fprintf(stderr, "recoverPose(): good points = %d / %d", ngood,  mask.rows);

  pmat(R, "R");
  pmat(T, "T");

  /*
   * Compute 'derotation' homography required to make optical axes parallel
   * */
  H = cameraMatrix * R.t() * cameraMatrix.inv();
  pmat(H, "H");


  /*
   * Warp second frame keypoints to 'derotate' and re-estimate fundamental matrix.
   * The resulting fundamental matrix shoud give the same coordinates of epipoles in both frames.
   * */

  cv::perspectiveTransform(matched_keypoints[1],
      matched_keypoints[1],
      H);

  F = cv::findFundamentalMat(matched_keypoints[0], matched_keypoints[1],
      mask,
      cv::FM_LMEDS,
      0.5,
      0.999);

  fprintf(stderr, "findFundamentalMat(): good points = %d / %d\n", cv::countNonZero(mask), mask.rows);
  pmat(F, "F2");


  /*
   * Compute the direction of key points movement - FROM or TOWARDS the epipoles.
   * The epipoles locations in pixels should coincide (be equal) on both frames if above homography and fundamental matrix was computed correctly.
   */
  c_polar_stereo_rectification::compute_epipoles(cv::Matx33d(F),
      &epipoles[0], &epipoles[1]);

  *output_median_radial_flow = median_radial_flow(epipoles[0], epipoles[1],
      matched_keypoints[0], matched_keypoints[1],
      mask);

  fprintf(stderr, "Final epipoles: E0={%+g %+g} E1={%+g %+g} median_radial_flow=%+g\n",
      epipoles[0].x, epipoles[0].y,
      epipoles[1].x, epipoles[1].y,
      *output_median_radial_flow);


  output_rotation_matrix = R;
  output_translation_vector = T;
  output_fundamental_matrix = F;
  output_homography1 = H;

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

  /* Stereo rectificator */
  c_polar_stereo_rectification rectification;


  /* Camera ID for camera matrix */
  enum camera_id {
    camera_unknown = -1,
    camera_kitti_0,
    camera_kitti_1,
    camera_kitti_2,
    camera_kitti_3,
    camera_india_f
  } camera = camera_unknown;


  /* Camera Matrix  */
  cv::Matx33d cameraMatrix;


  /*
   * Parse command line arguments
   * */
  for ( int i = 1; i < argc; ++i ) {

    if ( strcmp(argv[i], "--help") == 0 ) {

      fprintf(stdout,
          "Usage:\n"
          "    PolarRectification image1 image2 ARGS\n"
          "\n"
          "ARGS:\n"
          "  -c <camera-name>\n"
          "        Specify camera name: one of kitti0 kitti1 kitti2 kitti3 indiaf\n"
          "\n"

      );
      return 0;
    }


    if ( strcmp(argv[i], "-c") == 0 ) { // camera name
      if ( ++i >= argc ) {
        fprintf(stderr, "Command line error: No camera name specified after '%s' option\n",
            argv[i - 1]);
        return 1;
      }

      if ( strcasecmp(argv[i], "kitti0") == 0 ) {
        camera = camera_kitti_0;
      }
      else if ( strcasecmp(argv[i], "kitti1") == 0 ) {
        camera = camera_kitti_1;
      }
      else if ( strcasecmp(argv[i], "kitti2") == 0 ) {
        camera = camera_kitti_2;
      }
      else if ( strcasecmp(argv[i], "kitti3") == 0 ) {
        camera = camera_kitti_3;
      }
      else if ( strcasecmp(argv[i], "indiaf") == 0 ) {
        camera = camera_india_f;
      }
      else {
        fprintf(stderr, "Command line error: Invalid camera name '%s' specified\n",
            argv[i]);
        return 1;
      }
    }

    else if ( input_file_names[0].empty() ) {  /* first input file name */
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
   * Select camera matrix based on user-specified camera id
   */
  if ( camera == camera_unknown ) {
    fprintf(stderr, "No camera name specified in command line, "
        "can not select appropriate camera matrix\n");
    return 1;
  }


  if ( camera == camera_india_f ) {

    cameraMatrix = cv::Matx33d(
        3654.002995893292, 0, 929.8061151563577,
        0, 3685.505662347394, 640.5609325644043,
        0, 0, 1);

  }
  else {
      /* Data extracted from kitti/raw/2011_09_26/calib_cam_to_cam.txt */

    static double P_rect[4][3 * 4] = {
        /* P_rect_00: */
        {
            7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00,
            0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00
        },
        /* P_rect_01: */
        {
            7.215377e+02, 0.000000e+00, 6.095593e+02, -3.875744e+02,
            0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00
        },
        /* P_rect_02: */
        {
            7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01,
            0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01,
            0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03

        },
        /* P_rect_03: */
        {
            7.215377e+02, 0.000000e+00, 6.095593e+02, -3.395242e+02,
            0.000000e+00, 7.215377e+02, 1.728540e+02, 2.199936e+00,
            0.000000e+00, 0.000000e+00, 1.000000e+00, 2.729905e-03
        }
    };

    const int camera_index =
        camera - camera_kitti_0;

    cameraMatrix =
        cv::Mat1d(3, 4, P_rect[camera_index])(cv::Rect(0, 0, 3, 3));
  }

  pmat(cameraMatrix, "cameraMatrix");


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
   * Estimate fundamental matrix and 'derotation' homography required to make camera axes parallel
   */

  cv::Matx33d F; /* Fundamental matrix */
  cv::Matx33d R; /* Rotation matrix between frames */
  cv::Vec3d T; /* Translation vector between frames */
  cv::Matx33d H; /* 'Derotation' homography between frames */

  double median_radial_flow = 0; /* used to select between left/right frames for stereo matching */
  int c0 = 0, c1 = 1; /* Indexes of right / left frames, computed below based on the direction of camera movement */
  cv::Mat polar0, polar1; /* right / left rectified images */

  if ( !estimate_pose(input_images, cameraMatrix, R, T, F, H, &median_radial_flow) ) {
    fprintf(stderr, "estimate_pose() fails\n");
    return 1;
  }

  pmat(F, "F");
  pmat(R, "R");
  pmat(T, "T");

  /* select left / right image */
  if ( median_radial_flow >= 0 ) {
    c0 = 0;
    c1 = 1;
  }
  else {
    c0 = 1;
    c1 = 0;
  }
  fprintf(stderr, "median_radial_flow=%+g c0=%d c1=%d\n", median_radial_flow, c0, c1);


  /*
   * Apply 'derotation' homography to second image  and dump to disk
   * */
  cv::warpPerspective(input_images[1], input_images[1], H, input_images[1].size());
  cv::imwrite("warpPerspective1.png", input_images[1]);



  /*
   * Compute rectification mappings using fundamental maxtrix and image size provided
   * */
  if ( !rectification.compute(F, input_images[0].size()) ) {
    fprintf(stderr, "rectification.compute() fails\n");
    return 1;
  }

  fprintf(stderr, "epipole1: x=%g y=%g  epipole2: x=%g y=%g\n",
      rectification.epipole(0).x, rectification.epipole(0).y,
      rectification.epipole(1).x, rectification.epipole(1).y);



  /*
   * Remap (stereo rectify) input images
   * */
  rectification.remap(input_images[0], forward_transfrormed_images[0],
      input_images[1], forward_transfrormed_images[1]);


  /* Select which image is left and which is right for stereo matching */
  polar0 = forward_transfrormed_images[c0];
  polar1 = forward_transfrormed_images[c1];


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

  cv::imwrite("polar0.png",
      polar0);

  cv::imwrite("polar1.png",
      polar1);

  cv::imwrite("reverse0.png",
      reverse_transfrormed_images[0]);

  cv::imwrite("reverse1.png",
      reverse_transfrormed_images[1]);


  /*
   * Finish this test
   */
  return 0;
}

