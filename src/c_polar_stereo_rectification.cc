/*
 * c_polar_stereo_rectification.cc
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

#include "c_polar_stereo_rectification.h"


#ifndef CF_DEBUG
  #define CF_DEBUG(...) \
    fprintf(stderr, "%s() : %d ", __FUNCTION__, __LINE__), \
    fprintf(stderr, __VA_ARGS__), \
    fprintf(stderr, "\n")
#endif

#ifndef CF_ERROR
#define CF_ERROR  CF_DEBUG
#endif


template<class T>
static inline bool SIGN(T val)
{
  return (val >= 0);
}

template<class T>
static inline T distance(T x, T y)
{
  return std::sqrt(x * x + y * y);
}

template<class T>
static inline T squared_distance(T x, T y)
{
  return (x * x + y * y);
}

template<class T>
static inline bool IS_INSIDE_IMAGE(const cv::Point_<T> & point, const cv::Size & image_size)
{
  return (point.x >= 0) && (point.y >= 0) && (point.x < image_size.width) && (point.y < image_size.height);
}

template<class T>
static inline bool IS_INSIDE_IMAGE(const cv::Vec<T, 2> & point, const cv::Size & image_size)
{
  return (point[0] >= 0) && (point[1] >= 0) && (point[0] < image_size.width) && (point[1] < image_size.height);
}

// (py – qy)x + (qx – px)y + (pxqy – qxpy) = 0
template<class T>
static inline cv::Vec<T, 3> GET_LINE_FROM_POINTS(const cv::Point_<T> & point1, const cv::Point_<T> & point2)
{
  return cv::Vec<T, 3>(point1.y - point2.y, point2.x - point1.x, point1.x * point2.y - point2.x * point1.y);
}

bool c_polar_stereo_rectification::compute_epipoles(const cv::Matx33d & F,
    cv::Point2d * output_epipole0, cv::Point2d * output_epipole1)
{

#if 1

  try {
    cv::SVD svd(F);

    cv::Matx13d e1 = svd.vt.row(2);
    cv::Matx31d e2 = svd.u.col(2);

    output_epipole0->x = e1(0, 0) / e1(0, 2);
    output_epipole0->y = e1(0, 1) / e1(0, 2);

    output_epipole1->x = e2(0, 0) / e2(2, 0);
    output_epipole1->y = e2(1, 0) / e2(2, 0);

    return true;
  }
  catch (const std::exception & e) {
    CF_ERROR("cv::SVD() fails in %s(): %s", __func__,e.what());
  }
  catch (...) {
    CF_ERROR("cv::SVD() fails in %s()",__func__);
  }

  return false;

#else

  try {

    std::vector<double> eigenvalues;
    cv::Matx33d eigenvectors;

    cv::eigen(F.t() * F, eigenvalues, eigenvectors);
    output_epipole0->x = eigenvectors(2, 0) / eigenvectors(2, 2);
    output_epipole0->y = eigenvectors(2, 1) / eigenvectors(2, 2);

    cv::eigen(F * F.t(), eigenvalues, eigenvectors);
    output_epipole1->x = eigenvectors(2, 0) / eigenvectors(2, 2);
    output_epipole1->y = eigenvectors(2, 1) / eigenvectors(2, 2);

    return true;
  }
  catch (const std::exception & e) {
    CF_ERROR("cv::eigen() fails in %s(): %s", __func__,e.what());
  }
  catch (...) {
    CF_ERROR("cv::eigen() fails in %s()",__func__);
  }

  return false;

#endif
}

void c_polar_stereo_rectification::get_external_points(const cv::Point2d & epipole,
    const cv::Size & image_size,
    std::vector<cv::Point2d> & output_external_points)
{
  output_external_points.clear();

  if ( epipole.y < 0 ) {  // Cases 1, 2 and 3
    if ( epipole.x < 0 ) {  // Case 1
      output_external_points.resize(2);
      output_external_points[0] = cv::Point2d(image_size.width - 1, 0);
      output_external_points[1] = cv::Point2d(0, image_size.height - 1);
    }
    else if ( epipole.x <= image_size.width - 1 ) {  // Case 2
      output_external_points.resize(2);
      output_external_points[0] = cv::Point2d(image_size.width - 1, 0);
      output_external_points[1] = cv::Point2d(0, 0);
    }
    else {  // Case 3
      output_external_points.resize(2);
      output_external_points[0] = cv::Point2d(image_size.width - 1, image_size.height - 1);
      output_external_points[1] = cv::Point2d(0, 0);
    }
  }
  else if ( epipole.y <= image_size.height - 1 ) {  // Cases 4, 5 and 6
    if ( epipole.x < 0 ) {  // Case 4
      output_external_points.resize(2);
      output_external_points[0] = cv::Point2d(0, 0);
      output_external_points[1] = cv::Point2d(0, image_size.height - 1);
    }
    else if ( epipole.x <= image_size.width - 1 ) {  // Case 5
      output_external_points.resize(4);
      output_external_points[0] = cv::Point2d(0, 0);
      output_external_points[1] = cv::Point2d(image_size.width - 1, 0);
      output_external_points[2] = cv::Point2d(image_size.width - 1, image_size.height - 1);
      output_external_points[3] = cv::Point2d(0, image_size.height - 1);
    }
    else {  // Case 6
      output_external_points.resize(2);
      output_external_points[0] = cv::Point2d(image_size.width - 1, image_size.height - 1);
      output_external_points[1] = cv::Point2d(image_size.width - 1, 0);
    }
  }
  else {  // Cases 7, 8 and 9
    if ( epipole.x < 0 ) {  // Case 7
      output_external_points.resize(2);
      output_external_points[0] = cv::Point2d(0, 0);
      output_external_points[1] = cv::Point2d(image_size.width - 1, image_size.height - 1);
    }
    else if ( epipole.x <= image_size.width - 1 ) {  // Case 8
      output_external_points.resize(2);
      output_external_points[0] = cv::Point2d(0, image_size.height - 1);
      output_external_points[1] = cv::Point2d(image_size.width - 1, image_size.height - 1);
    }
    else {  // Case 9
      output_external_points.resize(2);
      output_external_points[0] = cv::Point2d(0, image_size.height - 1);
      output_external_points[1] = cv::Point2d(image_size.width - 1, 0);
    }
  }

}

void c_polar_stereo_rectification::estimate_rho_range(const cv::Point2d & epipole,
    const cv::Size & image_size,
    const std::vector<cv::Point2d> & external_points,
    double * output_min_rho, double * output_max_rho)
{
  if ( epipole.y < 0 ) {  // Cases 1, 2 and 3

    if ( epipole.x < 0 ) {  // Case 1

      *output_min_rho =  // Point A
          distance(epipole.x, epipole.y);

      *output_max_rho =  // Point D
          distance((image_size.width - 1) - epipole.x,
              (image_size.height - 1) - epipole.y);

    }
    else if ( epipole.x < image_size.width ) {  // Case 2

      *output_min_rho = -epipole.y;

      *output_max_rho = std::max(  // Point C, Point D
          distance(epipole.x,
              image_size.height - 1 - epipole.y),
          distance(image_size.width - 1 - epipole.x,
              image_size.height - 1 - epipole.y));

    }
    else {  // Case 3

      *output_min_rho =  // Point B
          distance(image_size.width - 1 - epipole.x,
              epipole.y);

      *output_max_rho =  // Point C
          distance(epipole.x,
              image_size.height - 1 - epipole.y);
    }
  }
  else if ( epipole.y < image_size.height ) {  // Cases 4, 5 and 6

    if ( epipole.x < 0 ) {  // Case 4

      *output_min_rho = -epipole.x;

      *output_max_rho = std::max(  // Point D, Point B
          distance(image_size.width - 1 - epipole.x,
              image_size.height - 1 - epipole.y),
          distance(image_size.width - 1 - epipole.x,
              epipole.y));

    }
    else if ( epipole.x < image_size.width ) {  // Case 5

      *output_min_rho = 0;

      *output_max_rho = std::max(  // Point A, Point C, Point D
          std::max(distance(epipole.x, epipole.y),
              distance(image_size.width - 1 - epipole.x,
                  epipole.y)),
          std::max(distance(epipole.x, image_size.height - 1 - epipole.y),
              distance(image_size.width - 1 - epipole.x,
                  image_size.height - 1 - epipole.y)));
    }
    else {  // Case 6

      *output_min_rho =
          epipole.x - (image_size.width - 1);

      *output_max_rho = std::max(  // Point A, Point C
          distance(epipole.x, epipole.y),
          distance(epipole.x, image_size.height - 1 - epipole.y));
    }
  }
  else {  // Cases 7, 8 and 9
    if ( epipole.x < 0 ) {  // Case 7

      *output_min_rho =  // Point C
          distance(epipole.x,
              image_size.height - 1 - epipole.y);

      *output_max_rho =  // Point B
          distance(image_size.width - 1 - epipole.x,
              epipole.y);

    }
    else if ( epipole.x < image_size.width ) {  // Case 8

      *output_min_rho =
          epipole.y - (image_size.height - 1);

      *output_max_rho = std::max(  // Point A, Point B
          distance(epipole.x, epipole.y),
          distance(image_size.width - 1 - epipole.x,
              epipole.y));
    }
    else {  // Case 9

      *output_min_rho =  // Point D
          distance(image_size.width - 1 - epipole.x,
              image_size.height - 1 - epipole.y);

      *output_max_rho =  // Point A
          distance(epipole.x, epipole.y);
    }
  }
}

void c_polar_stereo_rectification::compute_epilines(const std::vector<cv::Point2d> & points, int whichImage,
    const cv::Matx33d & F, const std::vector<cv::Vec3d> & oldlines,
    std::vector<cv::Vec3d> & newLines)
{
  cv::computeCorrespondEpilines(points, whichImage, F, newLines);

  for ( uint32_t i = 0; i < oldlines.size(); i++ ) {
    if ( (SIGN(oldlines[i][0]) != SIGN(newLines[i][0])) &&
        (SIGN(oldlines[i][1]) != SIGN(newLines[i][1])) ) {
      newLines[i] *= -1;
    }
  }
}

bool c_polar_stereo_rectification::line_intersects_segment(const cv::Vec3d & line,
    const cv::Point2d & p1, const cv::Point2d & p2,
    cv::Point2d * intersection)
{
  using std::min;
  using std::max;

  const cv::Vec3d segment = GET_LINE_FROM_POINTS(p1, p2);

  if ( intersection ) {

    *intersection = cv::Point2d(
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max());
  }

  // Lines are represented as ax + by + c = 0, so
  // y = -(ax+c)/b. If y1=y2, then we have to obtain x, which is
  // x = (b1 * c2 - b2 * c1) / (b2 * a1 - b1 * a2)
  if ( !(segment[1] * line[0] - line[1] * segment[0]) ) {
    return false;
  }

  const double x = (line[1] * segment[2] - segment[1] * line[2]) / (segment[1] * line[0] - line[1] * segment[0]);
  const int rx = cvRound(x);

  const double y = -(line[0] * x + line[2]) / line[1];
  const int ry = cvRound(y);

  if ( (rx >= (int) min(p1.x, p2.x)) && (rx <= (int) max(p1.x, p2.x)) ) {
    if ( (ry >= (int) min(p1.y, p2.y)) && (ry <= (int) max(p1.y, p2.y)) ) {
      if ( intersection ) {
        *intersection = cv::Point2d(x, y);
      }
      return true;
    }
  }

  return false;
}

bool c_polar_stereo_rectification::line_intersects_rect(const cv::Vec3d & line, const cv::Size & image_size,
    cv::Point2d * intersection)
{
  return line_intersects_segment(line, cv::Point2d(0, 0), cv::Point2d(image_size.width - 1, 0), intersection)
      ||
      line_intersects_segment(line, cv::Point2d(image_size.width - 1, 0),
          cv::Point2d(image_size.width - 1, image_size.height - 1), intersection)
      ||
      line_intersects_segment(line, cv::Point2d(image_size.width - 1, image_size.height - 1),
          cv::Point2d(0, image_size.height - 1), intersection)
      ||
      line_intersects_segment(line, cv::Point2d(0, image_size.height - 1),
          cv::Point2d(0, 0), intersection);
}

bool c_polar_stereo_rectification::is_the_right_point(const cv::Point2d & epipole,
    const cv::Point2d & intersection,
    const cv::Vec3d & line,
    const cv::Point2d * lastPoint)
{
  if ( lastPoint != NULL ) {

    cv::Vec3d v1(lastPoint->x - epipole.x, lastPoint->y - epipole.y, 0.0);
    v1 /= cv::norm(v1);

    cv::Vec3d v2(intersection.x - epipole.x, intersection.y - epipole.y, 0.0);
    v2 /= cv::norm(v2);

    if ( fabs(acos(v1.dot(v2))) <= CV_PI / 2.0 ) {
      return true;
    }
  }
  else {

    if ( (line[0] > 0) && (epipole.y < intersection.y) ) {
      return false;
    }
    if ( (line[0] < 0) && (epipole.y > intersection.y) ) {
      return false;
    }
    if ( (line[1] > 0) && (epipole.x > intersection.x) ) {
      return false;
    }
    if ( (line[1] < 0) && (epipole.x < intersection.x) ) {
      return false;
    }

    return true;
  }

  return false;
}

cv::Point2d c_polar_stereo_rectification::get_border_intersection(const cv::Point2d & epipole, const cv::Vec3d & line,
    const cv::Size & image_size,
    const cv::Point2d * lastPoint)
{

  cv::Point2d intersection(-1, -1);

  if ( IS_INSIDE_IMAGE(epipole, image_size) ) {
    if ( line_intersects_segment(line, cv::Point2d(0, 0), cv::Point2d(image_size.width - 1, 0), &intersection) ) {
      if ( is_the_right_point(epipole, intersection, line, lastPoint) ) {

        return intersection;
      }
    }
    if ( line_intersects_segment(line, cv::Point2d(image_size.width - 1, 0),
        cv::Point2d(image_size.width - 1, image_size.height - 1), &intersection) ) {
      if ( is_the_right_point(epipole, intersection, line, lastPoint) ) {

        return intersection;
      }
    }
    if ( line_intersects_segment(line, cv::Point2d(image_size.width - 1, image_size.height - 1),
        cv::Point2d(0, image_size.height - 1), &intersection) ) {
      if ( is_the_right_point(epipole, intersection, line, lastPoint) ) {

        return intersection;
      }
    }
    if ( line_intersects_segment(line, cv::Point2d(0, image_size.height - 1), cv::Point2d(0, 0), &intersection) ) {
      if ( is_the_right_point(epipole, intersection, line, lastPoint) ) {

        return intersection;
      }
    }
  }
  else {
    double maxDist = std::numeric_limits<double>::min();
    cv::Point2d tmpIntersection(-1, -1);

    if ( line_intersects_segment(line, cv::Point2d(0, 0), cv::Point2d(image_size.width - 1, 0), &tmpIntersection) ) {

      const double dist2 =
          squared_distance(tmpIntersection.x - epipole.x,
              tmpIntersection.x - epipole.x);

      if ( dist2 > maxDist ) {
        maxDist = dist2;
        intersection = tmpIntersection;
      }
    }
    if ( line_intersects_segment(line, cv::Point2d(image_size.width - 1, 0),
        cv::Point2d(image_size.width - 1, image_size.height - 1), &tmpIntersection) ) {

      const double dist2 =
          squared_distance(tmpIntersection.x - epipole.x,
              tmpIntersection.x - epipole.x);

      if ( dist2 > maxDist ) {
        maxDist = dist2;
        intersection = tmpIntersection;
      }
    }
    if ( line_intersects_segment(line, cv::Point2d(image_size.width - 1, image_size.height - 1),
        cv::Point2d(0, image_size.height - 1), &tmpIntersection) ) {

      const double dist2 =
          squared_distance(tmpIntersection.x - epipole.x,
              tmpIntersection.x - epipole.x);

      if ( dist2 > maxDist ) {
        maxDist = dist2;
        intersection = tmpIntersection;
      }
    }
    if ( line_intersects_segment(line, cv::Point2d(0, image_size.height - 1), cv::Point2d(0, 0), &tmpIntersection) ) {

      const double dist2 =
          squared_distance(tmpIntersection.x - epipole.x,
              tmpIntersection.x - epipole.x);

      if ( dist2 > maxDist ) {
        maxDist = dist2;
        intersection = tmpIntersection;
      }
    }
  }

  return intersection;
}

void c_polar_stereo_rectification::get_border_intersections(const cv::Point2d & epipole, const cv::Vec3d & line,
    const cv::Size & image_size,
    std::vector<cv::Point2d> & output_intersections)
{
  cv::Point2d intersection(-1, -1);

  output_intersections.clear();
  output_intersections.reserve(2);

  if ( line_intersects_segment(line, cv::Point2d(0, 0), cv::Point2d(image_size.width - 1, 0), &intersection) ) {
    output_intersections.emplace_back(intersection);
  }
  if ( line_intersects_segment(line, cv::Point2d(image_size.width - 1, 0),
      cv::Point2d(image_size.width - 1, image_size.height - 1), &intersection) ) {
    output_intersections.emplace_back(intersection);
  }
  if ( line_intersects_segment(line, cv::Point2d(image_size.width - 1, image_size.height - 1),
      cv::Point2d(0, image_size.height - 1), &intersection) ) {
    output_intersections.emplace_back(intersection);
  }
  if ( line_intersects_segment(line, cv::Point2d(0, image_size.height - 1), cv::Point2d(0, 0), &intersection) ) {
    output_intersections.emplace_back(intersection);
  }
}

cv::Point2d c_polar_stereo_rectification::get_nearest_intersection(const cv::Point2d& oldEpipole,
    const cv::Point2d& newEpipole,
    const cv::Vec3d& line,
    const cv::Point2d& oldPoint,
    const cv::Size & image_size)
{
  std::vector<cv::Point2d> intersections;
  get_border_intersections(newEpipole, line, image_size, intersections);

  double minAngle = std::numeric_limits<double>::max();
  cv::Point2d point(-1, -1);

  cv::Vec3d v1(oldPoint.x - oldEpipole.x, oldPoint.y - oldEpipole.y, 0.0);
  v1 /= cv::norm(v1);

  for ( uint32_t i = 0; i < intersections.size(); i++ ) {
    cv::Vec3d v(intersections[i].x - newEpipole.x, intersections[i].y - newEpipole.y, 0.0);
    v /= cv::norm(v);

    const double & angle = fabs(acos(v.dot(v1)));

    if ( angle < minAngle ) {
      minAngle = angle;
      point = intersections[i];
    }
  }

  return point;
}

void c_polar_stereo_rectification::estimate_common_region(const cv::Size & image_size)
{
  std::vector<cv::Point2d> external_points[2];

  for ( int i = 0; i < 2; ++i ) {

    get_external_points(epipoles[i],
        image_size,
        external_points[i]);

    estimate_rho_range(epipoles[i],
        image_size,
        external_points[i],
        &minRho[i],
        &maxRho[i]);
  }

  if ( !IS_INSIDE_IMAGE(epipoles[0], image_size) && !IS_INSIDE_IMAGE(epipoles[1], image_size) ) {
    // CASE 1: Both outside
    const cv::Vec3d line11 = GET_LINE_FROM_POINTS(epipoles[0], external_points[0][0]);
    const cv::Vec3d line12 = GET_LINE_FROM_POINTS(epipoles[0], external_points[0][1]);

    const cv::Vec3d line23 = GET_LINE_FROM_POINTS(epipoles[1], external_points[1][0]);
    const cv::Vec3d line24 = GET_LINE_FROM_POINTS(epipoles[1], external_points[1][1]);

    std::vector<cv::Vec3d> inputLines(2), outputLines;
    inputLines[0] = line23;
    inputLines[1] = line24;
    compute_epilines(external_points[1], 2, F, inputLines, outputLines);
    const cv::Vec3d line13 = outputLines[0];
    const cv::Vec3d line14 = outputLines[1];

    inputLines[0] = line11;
    inputLines[1] = line12;
    compute_epilines(external_points[0], 1, F, inputLines, outputLines);
    const cv::Vec3d line21 = outputLines[0];
    const cv::Vec3d line22 = outputLines[1];

    // Beginning and ending lines
    m_lineB[0] = line_intersects_rect(line13, image_size) ? line13 : line11;
    m_lineE[0] = line_intersects_rect(line14, image_size) ? line14 : line12;
    m_lineB[1] = line_intersects_rect(line21, image_size) ? line21 : line23;
    m_lineE[1] = line_intersects_rect(line22, image_size) ? line22 : line24;

    // Beginning and ending lines intersection with the borders
    std::vector<cv::Point2d> intersections;
    double maxDist;

    //
    get_border_intersections(epipoles[0], m_lineB[0], image_size, intersections);
    maxDist = std::numeric_limits<double>::min();
    for ( uint i = 0, n = intersections.size(); i < n; ++i ) {
      const double dist = cv::norm(epipoles[0] - intersections[i]);
      if ( dist > maxDist ) {
        maxDist = dist;
        m_b[0] = intersections[i];
      }
    }

    //
    get_border_intersections(epipoles[1], m_lineB[1], image_size, intersections);
    maxDist = std::numeric_limits<double>::min();
    for ( uint i = 0, n = intersections.size(); i < n; ++i ) {
      const double dist = cv::norm(epipoles[1] - intersections[i]);
      if ( dist > maxDist ) {
        maxDist = dist;
        m_b[1] = intersections[i];
      }
    }

    //
    get_border_intersections(epipoles[0], m_lineE[0], image_size, intersections);
    maxDist = std::numeric_limits<double>::min();
    for ( uint i = 0, n = intersections.size(); i < n; ++i ) {
      const double dist = cv::norm(epipoles[0] - intersections[i]);
      if ( dist > maxDist ) {
        maxDist = dist;
        m_e[0] = intersections[i];
      }
    }

    //
    get_border_intersections(epipoles[1], m_lineE[1], image_size, intersections);
    maxDist = std::numeric_limits<double>::min();
    for ( uint i = 0, n = intersections.size(); i < n; ++i ) {
      const double dist = cv::norm(epipoles[1] - intersections[i]);
      if ( dist > maxDist ) {
        maxDist = dist;
        m_e[0] = intersections[i];
      }
    }

  }
  else if ( IS_INSIDE_IMAGE(epipoles[0], image_size) && IS_INSIDE_IMAGE(epipoles[1], image_size) ) {
    // CASE 2: Both inside

    m_lineB[0] = GET_LINE_FROM_POINTS(epipoles[0], external_points[0][0]);
    m_lineE[0] = m_lineB[0];

    std::vector<cv::Vec3d> inputLines(1), outputLines;
    inputLines[0] = m_lineB[0];
    compute_epilines(external_points[0], 1, F, inputLines, outputLines);

    m_lineB[1] = outputLines[0];
    m_lineE[1] = outputLines[0];

    m_b[0] = get_border_intersection(epipoles[0], m_lineB[0], image_size);
    m_e[0] = get_border_intersection(epipoles[0], m_lineE[0], image_size);

    m_b[1] = m_e[1] = get_nearest_intersection(epipoles[0], epipoles[1], m_lineB[1], m_b[0], image_size);

  }
  else {
    // CASE 3: One inside and one outside
    if ( IS_INSIDE_IMAGE(epipoles[0], image_size) ) {
      // CASE 3.1: Only the first epipole is inside

      const cv::Vec3d line23 = GET_LINE_FROM_POINTS(epipoles[1], external_points[1][0]);
      const cv::Vec3d line24 = GET_LINE_FROM_POINTS(epipoles[1], external_points[1][1]);

      std::vector<cv::Vec3d> inputLines(2), outputLines;
      inputLines[0] = line23;
      inputLines[1] = line24;
      compute_epilines(external_points[1], 2, F, inputLines, outputLines);
      const cv::Vec3d & line13 = outputLines[0];
      const cv::Vec3d & line14 = outputLines[1];

      m_lineB[0] = line13;
      m_lineE[0] = line14;
      m_lineB[1] = line23;
      m_lineE[1] = line24;

      m_b[1] = get_border_intersection(epipoles[1], m_lineB[1], image_size);
      m_e[1] = get_border_intersection(epipoles[1], m_lineE[1], image_size);

      m_b[0] = get_nearest_intersection(epipoles[1], epipoles[0], m_lineB[0], m_b[1], image_size);
      m_e[0] = get_nearest_intersection(epipoles[1], epipoles[0], m_lineE[0], m_e[1], image_size);

    }
    else {
      // CASE 3.2: Only the second epipole is inside
      const cv::Vec3d line11 = GET_LINE_FROM_POINTS(epipoles[0], external_points[0][0]);
      const cv::Vec3d line12 = GET_LINE_FROM_POINTS(epipoles[0], external_points[0][1]);

      std::vector<cv::Vec3d> inputLines(2), outputLines;
      inputLines[0] = line11;
      inputLines[1] = line12;
      compute_epilines(external_points[0], 1, F, inputLines, outputLines);
      const cv::Vec3d & line21 = outputLines[0];
      const cv::Vec3d & line22 = outputLines[1];

      m_lineB[0] = line11;
      m_lineE[0] = line12;
      m_lineB[1] = line21;
      m_lineE[1] = line22;

      m_b[0] = get_border_intersection(epipoles[0], m_lineB[0], image_size);
      m_e[0] = get_border_intersection(epipoles[0], m_lineE[0], image_size);

      m_b[1] = get_nearest_intersection(epipoles[0], epipoles[1], m_lineB[1], m_b[0], image_size);
      m_e[1] = get_nearest_intersection(epipoles[0], epipoles[1], m_lineE[1], m_e[0], image_size);
    }
  }
}


void c_polar_stereo_rectification::get_new_point_and_line_single_image(const cv::Size & image_size, int whichImage,
    const cv::Point2d & pOld1, const cv::Point2d & pOld2,
    cv::Point2d * pNew1, cv::Vec3d * newLine1,
    cv::Point2d * pNew2, cv::Vec3d * newLine2)  const
{

  // We obtain vector v
  cv::Vec2d v;

  cv::Vec3d vBegin(m_b[0].x - epipoles[0].x, m_b[0].y - epipoles[0].y, 0.0);
  cv::Vec3d vCurr(pOld1.x - epipoles[0].x, pOld1.y - epipoles[0].y, 0.0);
  cv::Vec3d vEnd(m_e[0].x - epipoles[0].x, m_e[0].y - epipoles[0].y, 0.0);

  vBegin /= cv::norm(vBegin);
  vCurr /= cv::norm(vCurr);
  vEnd /= cv::norm(vEnd);

  if ( IS_INSIDE_IMAGE(epipoles[0], image_size) ) {
    if ( IS_INSIDE_IMAGE(epipoles[1], image_size) ) {
      v = cv::Vec2d(vCurr[1], -vCurr[0]);
    }
    else {
      vBegin = cv::Vec3d(m_b[1].x - epipoles[1].x, m_b[1].y - epipoles[1].y, 0.0);
      vCurr = cv::Vec3d(pOld2.x - epipoles[1].x, pOld2.y - epipoles[1].y, 0.0);
      vEnd = cv::Vec3d(m_e[1].x - epipoles[1].x, m_e[1].y - epipoles[1].y, 0.0);

      vBegin /= cv::norm(vBegin);
      vCurr /= cv::norm(vCurr);
      vEnd /= cv::norm(vEnd);

      const cv::Vec3d vCross = vBegin.cross(vEnd);

      v = cv::Vec2d(vCurr[1], -vCurr[0]);
      if ( vCross[2] > 0.0 ) {
        v = -v;
      }
    }
  }
  else {
    const cv::Vec3d vCross = vBegin.cross(vEnd);

    v = cv::Vec2d(vCurr[1], -vCurr[0]);
    if ( vCross[2] > 0.0 ) {
      v = -v;
    }
  }

  *pNew1 = cv::Point2d(pOld1.x + v[0] * m_stepSize, pOld1.y + v[1] * m_stepSize);
  *newLine1 = GET_LINE_FROM_POINTS(epipoles[0], *pNew1);

  if ( !IS_INSIDE_IMAGE(epipoles[0], image_size) ) {
    *pNew1 = get_border_intersection(epipoles[0], *newLine1, image_size, &pOld1);
  }
  else {
    *pNew1 = get_nearest_intersection(epipoles[0], epipoles[0], *newLine1, pOld1, image_size);
  }

  std::vector<cv::Point2d> points(1);
  points[0] = *pNew1;
  std::vector<cv::Vec3d> inLines(1);
  inLines[0] = *newLine1;
  std::vector<cv::Vec3d> outLines(1);
  compute_epilines(points, whichImage, F, inLines, outLines);
  *newLine2 = outLines[0];

  if ( !IS_INSIDE_IMAGE(epipoles[1], image_size) ) {
    cv::Point2d tmpPoint = get_border_intersection(epipoles[1], *newLine2, image_size, &pOld2);
    *pNew2 = tmpPoint;
  }
  else {
    std::vector<cv::Point2d> intersections;
    get_border_intersections(epipoles[1], *newLine2, image_size, intersections);
    *pNew2 = intersections[0];

    double minDist = std::numeric_limits<double>::max();
    for ( uint32_t i = 0; i < intersections.size(); i++ ) {
      const double dist = (pOld2.x - intersections[i].x) * (pOld2.x - intersections[i].x) +
          (pOld2.y - intersections[i].y) * (pOld2.y - intersections[i].y);
      if ( minDist > dist ) {
        minDist = dist;
        *pNew2 = intersections[i];
      }
    }
  }

}

void c_polar_stereo_rectification::get_new_epiline(const cv::Size & image_size,
    const cv::Point2d pOld1, const cv::Point2d pOld2,
    cv::Point2d * pNew1, cv::Point2d * pNew2,
    cv::Vec3d * newLine1, cv::Vec3d * newLine2) const
{
  get_new_point_and_line_single_image(image_size, 1, pOld1, pOld2, pNew1, newLine1, pNew2, newLine2);
}



void c_polar_stereo_rectification::compute_transformation_points(const cv::Size & image_size)
{
  cv::Point2d p1 = m_b[0], p2 = m_b[1];
  cv::Vec3d line1 = m_lineB[0], line2 = m_lineB[1];

  for ( int i = 0; i < 2; ++i ) {
    theta_points[i].clear();
    theta_points[i].reserve(2 * (image_size.width + image_size.height));
  }

  int32_t crossesLeft = 0;
  if ( IS_INSIDE_IMAGE(epipoles[0], image_size) && IS_INSIDE_IMAGE(epipoles[1], image_size) )
    crossesLeft++;

  uint32_t thetaIdx = 0;
  double lastCrossProd = 0;

  while ( true ) {

    theta_points[0].push_back(p1);
    theta_points[1].push_back(p2);

    cv::Vec3d v0(p1.x - epipoles[0].x, p1.y - epipoles[0].y, 1.0);
    v0 /= cv::norm(v0);
    cv::Point2d oldP1 = p1;

    get_new_epiline(image_size, p1, p2, &p1, &p2, &line1, &line2);

    // Check if we reached the end
    cv::Vec3d v1(p1.x - epipoles[0].x, p1.y - epipoles[0].y, 0.0);
    v1 /= cv::norm(v1);

    cv::Vec3d v2(m_e[0].x - epipoles[0].x, m_e[0].y - epipoles[0].y, 0.0);
    v2 /= cv::norm(v2);

    cv::Vec3d v3(oldP1.x - epipoles[0].x, oldP1.y - epipoles[0].y, 0.0);
    v3 /= cv::norm(v3);

    const double crossProd = v1.cross(v2)[2];

    if ( thetaIdx != 0 ) {
      if ( (SIGN(lastCrossProd) != SIGN(crossProd)) || (fabs(acos(v1.dot(-v3))) < 0.01) || (p1 == cv::Point2d(-1, -1)) )
        crossesLeft--;

      if ( (crossesLeft < 0) ) {
        break;
      }
    }
    lastCrossProd = crossProd;
    thetaIdx++;

  }
  theta_points[0].pop_back();
  theta_points[1].pop_back();
}

void c_polar_stereo_rectification::compute_remap(const cv::Size & image_size,
    const cv::Point2d & epipole,
    const cv::Point2d & p2,
    int thetaIdx, double minRho, double maxRho,
    cv::Mat2f & rmap, cv::Mat2f & imap)
{
  cv::Vec2d v(p2.x - epipole.x, p2.y - epipole.y);

  const double maxDist = cv::norm(v);
  v /= maxDist;

  uint rhoIdx = 0;

  for ( double rho = minRho, maxrho = std::min(maxDist, maxRho); rho <= maxrho; rho += 1.0, ++rhoIdx ) {

    const cv::Vec2f target(
        v[0] * rho + epipole.x,   // x
        v[1] * rho + epipole.y);  // y

    if ( IS_INSIDE_IMAGE(target, image_size) ) {

      const int xx = std::max(0, std::min(image_size.width - 1, cvRound(target[0])));
      const int yy = std::max(0, std::min(image_size.height - 1, cvRound(target[1])));

      rmap[thetaIdx][rhoIdx] = target;
      imap[yy][xx][0] = rhoIdx;
      imap[yy][xx][1] = thetaIdx;

    }
  }
}



void c_polar_stereo_rectification::build_remaps(const cv::Size & image_size)
{
  const double rhoRange =
      std::max(maxRho[0] - minRho[0], maxRho[1] - minRho[1]) + 1;

  for ( int i = 0; i < 2; ++i ) {
    rmaps[i].create(theta_points[i].size(), rhoRange), rmaps[i].setTo(-1);
    imaps[i].create(image_size), imaps[i].setTo(-1);
  }

  for ( int i = 0; i < 2; ++i ) {
    for ( uint thetaIdx = 0, n = theta_points[i].size(); thetaIdx < n; ++thetaIdx ) {
      compute_remap(image_size,
          epipoles[i],
          theta_points[i][thetaIdx],
          thetaIdx,
          minRho[i],
          maxRho[i],
          rmaps[i],
          imaps[i]);
    }
  }
}


/**
 * @brief Compute the forward and inverse mappings based on provided fundamental matrix
 * */
bool c_polar_stereo_rectification::compute(const cv::Matx33d & input_fundamental_matrix, const cv::Size & image_size)
{
  this->F = input_fundamental_matrix;

  if ( !compute_epipoles(F, &epipoles[0], &epipoles[1]) ) {
    CF_ERROR("compute_epipoles() fails in %s()", __func__);
    return false;
  }

  if ( SIGN(epipoles[0].x) != SIGN(epipoles[1].x) && SIGN(epipoles[0].y) != SIGN(epipoles[1].y) ) {
    epipoles[1] *= -1;
  }

  estimate_common_region(image_size);
  compute_transformation_points(image_size);
  build_remaps(image_size);

  for ( int i = 0; i < 2; ++i ) {
    theta_points[i].clear();
    // theta_points_[i].shrink_to_fit();
  }

  return true;
}


/**
 * @brief Example of usage of forwad mapping
 * */
void c_polar_stereo_rectification::remap(cv::InputArray src1, cv::OutputArray dst1,
    cv::InputArray src2, cv::OutputArray dst2,
    int interpolation,
    int border_mode,
    const cv::Scalar & border_value) const
{
  if ( !src1.empty() && dst1.needed() ) {

    cv::remap(src1, dst1,
        forward_map(0),
        cv::noArray(),
        interpolation,
        border_mode,
        border_value);

  }

  if ( !src2.empty() && dst2.needed() ) {

    cv::remap(src2, dst2,
        forward_map(1),
        cv::noArray(),
        interpolation,
        border_mode,
        border_value);

  }
}


/**
 * @brief Example of usage of reverse mapping
 * */
void c_polar_stereo_rectification::unmap(cv::InputArray src1, cv::OutputArray dst1,
    cv::InputArray src2, cv::OutputArray dst2,
    int interpolation,
    int border_mode,
    const cv::Scalar & border_value) const
{
  if ( !src1.empty() && dst1.needed() ) {

    cv::remap(src1, dst1,
        reverse_map(0),
        cv::noArray(),
        interpolation,
        border_mode,
        border_value);

  }

  if ( !src2.empty() && dst2.needed() ) {

    cv::remap(src2, dst2,
        reverse_map(1),
        cv::noArray(),
        interpolation,
        border_mode,
        border_value);
  }
}

/**
 * @brief  Read-Only access to stored copy of input fundamental matrix
 * */
const cv::Matx33d & c_polar_stereo_rectification::fundamental_matrix() const
{
  return this->F;
}

/**
 * @brief Read-Only access to epipoles computed for each image index [0..1]
 * */
const cv::Point2d & c_polar_stereo_rectification::epipole(int index) const
{
  return this->epipoles[index];
}

/**
 * @brief Read-Only access to computed forward mapping computed for each image index [0..1]
 *  Use this map as argument for cv::remap() for forward (rectification) mapping.
 *  See this_class::remap() for example of usage.
 * */
const cv::Mat2f & c_polar_stereo_rectification::forward_map(int index) const
{
  return this->rmaps[index];
}

/**
 * @brief Read-Only access to computed reverse mapping computed for each image index [0..1]
 *  Use this map as argument for cv::remap() for reverse mapping.
 *  See this_class::unmap() for example of usage.
 * */
const cv::Mat2f & c_polar_stereo_rectification::reverse_map(int index) const
{
  return this->imaps[index];
}
