/*
 * c_polar_stereo_rectification.h
 *
 *  Created on: Dec 20, 2020
 *      Author: amyznikov
 *
 * This code is heavily based on the Néstor Morales Hernández 'PolarCalibration' project
 *    https://github.com/nestormh/PolarCalibration
 *
 * Implementation of paper
 *    "M. Pollefeys, R. Koch and L. Van Gool, A simple and efficient rectification method for general motion",
 *    http://www.inf.ethz.ch/personal/pomarc/pubs/PollefeysICCV99.pdf
 */

#ifndef __c_polar_stereo_rectification_h__
#define __c_polar_stereo_rectification_h__

#include <opencv2/opencv.hpp>

/** @brief Polar stereo rectification for general motion
 *
 * This code is based on the Néstor Morales Hernández 'PolarCalibration' project
 *    https://github.com/nestormh/PolarCalibration
 *
 * Implementation of paper
 *    "M. Pollefeys, R. Koch and L. Van Gool, A simple and efficient rectification method for general motion",
 *    http://www.inf.ethz.ch/personal/pomarc/pubs/PollefeysICCV99.pdf
 *
 */
class c_polar_stereo_rectification
{
public: // public interface

  typedef c_polar_stereo_rectification
      this_class;


  /**
   * @brief Compute the forward and inverse mappings based on provided fundamental matrix
   * */
  bool compute(const cv::Matx33d & input_fundamental_matrix,
      const cv::Size & image_size);

  /**
   * @brief  Read-Only access to stored copy of input fundamental matrix
   * */
  const cv::Matx33d & fundamental_matrix() const;

  /**
   * @brief Read-Only access to epipoles computed for each image index [0..1]
   * */
  const cv::Point2d & epipole (int index) const;

  /**
   * @brief Read-Only access to computed forward mapping computed for each image index [0..1]
   *  Use this map as argument for cv::remap() for forward (rectification) mapping.
   *  See this_class::remap() for example of usage.
   * */
  const cv::Mat2f & forward_map(int index) const;

  /**
   * @brief Read-Only access to computed reverse mapping computed for each image index [0..1]
   *  Use this map as argument for cv::remap() for reverse mapping.
   *  See this_class::unmap() for example of usage.
   * */
  const cv::Mat2f & reverse_map(int index) const;


  /**
   * @brief Example of usage of forwad mapping
   * */
  void remap(cv::InputArray src1, cv::OutputArray dst1,
      cv::InputArray src2, cv::OutputArray dst2) const;

  /**
   * @brief Example of usage of reverse mapping
   * */
  void unmap(cv::InputArray src1, cv::OutputArray dst1,
      cv::InputArray src2, cv::OutputArray dst2) const;

public: // utility subroutines, not a real part of public interface

  static bool compute_epipoles(const cv::Matx33d & F,
      cv::Point2d * output_epipole0,
      cv::Point2d * output_epipole1 );

  static void get_external_points(const cv::Point2d & epipole,
      const cv::Size & image_size,
      std::vector<cv::Point2d> & output_external_points);

  static void estimate_rho_range(const cv::Point2d &epipole,
      const cv::Size & image_size,
      const std::vector<cv::Point2d> & external_points,
      double * output_min_rho, double * output_max_rho);

  static void compute_epilines(const std::vector<cv::Point2d> & points,
      int whichImage,
      const cv::Matx33d & F,
      const std::vector <cv::Vec3d> & oldlines,
      std::vector <cv::Vec3d> & output_newLines);

  static bool line_intersects_segment(const cv::Vec3d & line,
      const cv::Point2d & p1,
      const cv::Point2d & p2,
      cv::Point2d * output_intersection = nullptr);

  static bool line_intersects_rect(const cv::Vec3d & line,
      const cv::Size & image_size,
      cv::Point2d * output_intersection = nullptr);

  static bool is_the_right_point(const cv::Point2d & epipole,
      const cv::Point2d & intersection,
      const cv::Vec3d & line,
      const cv::Point2d * lastPoint);

  static cv::Point2d get_border_intersection(const cv::Point2d & epipole,
      const cv::Vec3d & line,
      const cv::Size & image_size,
      const cv::Point2d * lastPoint = nullptr);

  static void get_border_intersections(const cv::Point2d & epipole, const cv::Vec3d & line,
      const cv::Size & image_size,
      std::vector<cv::Point2d> & output_intersections);

  static cv::Point2d get_nearest_intersection(const cv::Point2d& oldEpipole,
      const cv::Point2d& newEpipole,
      const cv::Vec3d& line,
      const cv::Point2d& oldPoint,
      const cv::Size & image_size);


protected: // protected utility routines

  void get_new_point_and_line_single_image(const cv::Size & image_size, int whichImage,
      const cv::Point2d & pOld1, const cv::Point2d & pOld2,
      cv::Point2d * pNew1, cv::Vec3d * newLine1,
      cv::Point2d * pNew2, cv::Vec3d * newLine2) const;

  void get_new_epiline(const cv::Size & image_size,
      const cv::Point2d pOld1, const cv::Point2d pOld2,
      cv::Point2d * pNew1, cv::Point2d * pNew2,
      cv::Vec3d * newLine1, cv::Vec3d * newLine2)  const;

  void estimate_common_region(const cv::Size & image_size);

  void compute_transformation_points(const cv::Size & image_size);

  static void compute_remap(const cv::Size & image_size,
      const cv::Point2d & epipole,
      const cv::Point2d & p2,
      int thetaIdx, double minRho, double maxRho,
      cv::Mat2f & map, cv::Mat2f & imap);

  void build_remaps(const cv::Size & image_size);

protected:

  /**@brief Lines step */
  const double m_stepSize = 1;

  /**@brief Fundamental Matrix */
  cv::Matx33d F;

  /**@brief Epipoles locations */
  cv::Point2d epipoles[2];

  /**@brief Estimated common region data */
  double minRho[2] = { 0, 0 }, maxRho[2] = { 0, 0 };
  cv::Vec3d m_lineB[2], m_lineE[2];
  cv::Point2d m_b[2], m_e[2];

  /**@brief Estimated transformation data */
  std::vector<cv::Point2d> theta_points[2];

  /**@brief Forward mappings */
  cv::Mat2f rmaps[2];

  /**@brief Reverse mappings */
  cv::Mat2f imaps[2];
};

#endif /* __c_polar_stereo_rectification_h__ */
