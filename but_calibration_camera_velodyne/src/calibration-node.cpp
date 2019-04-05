#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <algorithm>

#include "opencv2/opencv.hpp"

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <camera_info_manager/camera_info_manager.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl_ros/point_cloud.h>
#include <boost/foreach.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <velodyne_pointcloud/point_types.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>

#include <but_calibration_camera_velodyne/Velodyne.h>
#include <but_calibration_camera_velodyne/Calibration.h>
#include <but_calibration_camera_velodyne/Calibration3DMarker.h>
#include <but_calibration_camera_velodyne/Image.h>

#include <opencv2/core/eigen.hpp>

using namespace cv;
using namespace std;
using namespace ros;
using namespace message_filters;
using namespace pcl;
using namespace but_calibration_camera_velodyne;

string CAMERA_FRAME_TOPIC;
string CAMERA_INFO_TOPIC;
string VELODYNE_TOPIC;

// marker properties:
double STRAIGHT_DISTANCE; // 23cm
double RADIUS; // 8.25cm

Mat projection_matrix;
Mat frame_rgb;
Velodyne::Velodyne pointcloud;
cv::Mat cameraMatrix(3, 3, cv::DataType<double>::type);
cv::Mat distCoeffs(4,1,cv::DataType<double>::type);
bool doRefinement = false;

bool writeAllInputs()
{
  bool result = true;

  pointcloud.save("velodyne_pc.pcd");
  cv::imwrite("frame_rgb.png", frame_rgb);
  cv::FileStorage fs_P("projection.yml", cv::FileStorage::WRITE);
  fs_P << "P" << projection_matrix;
  fs_P.release();

  return result;
}

Calibration6DoF calibration(bool doRefinement = false)
{
  Mat frame_gray;
  cvtColor(frame_rgb, frame_gray, CV_BGR2GRAY);

  // Marker detection:
  Calibration3DMarker marker(frame_gray, projection_matrix, pointcloud.getPointCloud(), STRAIGHT_DISTANCE, RADIUS);
  vector<float> radii2D;
  vector<Point2f> centers2D;
  if (!marker.detectCirclesInImage(centers2D, radii2D))
  {
    return Calibration6DoF::wrong();
  }
  float radius2D = accumulate(radii2D.begin(), radii2D.end(), 0.0) / radii2D.size();
  vector<float> radii3D;
  vector<Point3f> centers3D;
  if (!marker.detectCirclesInPointCloud(centers3D, radii3D))
  {
    return Calibration6DoF::wrong();
  }
  float radius3D = accumulate(radii3D.begin(), radii3D.end(), 0.0) / radii3D.size();

  cv::Mat rvec(3,1,cv::DataType<double>::type);
  cv::Mat tvec(3,1,cv::DataType<double>::type);

  cv::solvePnP(centers3D, centers2D, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);

  cv::Mat C_R_L;
  cv::Rodrigues(rvec, C_R_L);
  Eigen::Matrix3f c_R_l;
  cv::cv2eigen(C_R_L, c_R_l);

  Eigen::Vector3f euler_angles = c_R_l.eulerAngles(2, 1, 0);
  std::cout << tvec << std::endl;
  std::cout << euler_angles << std::endl;
  for(int i=0;i<centers3D.size();i++){
    std::cout << centers3D[i].x << "\t" << centers3D[i].y << "\t" << centers3D[i].z << std::endl;
  }
//  std::vector<cv::Point2f> imagePoints;
//  cv::projectPoints(centers3D, rvec, tvec, cameraMatrix, distCoeffs, imagePoints, cv::noArray(), 0);
//  for(int i=0;i<centers2D.size();i++){
//    std::cout << centers2D[i].x << "\t" << centers2D[i].y << std::endl;
//    cv::circle(frame_rgb, centers2D[i], 16, CV_RGB(255, 0, 0), -1, 8, 0);
//    cv::circle(frame_rgb, imagePoints[i], 16, CV_RGB(0, 255, 0), -1, 8, 0);
//  }
//  std::cout << std::endl;
//
//  for(int i=0;i<imagePoints.size();i++){
//    std::cout << imagePoints[i].x << "\t" << imagePoints[i].y << std::endl;
//  }
//  std::cout << std::endl;
//  cv::Mat frame_resized;
//  cv::resize(frame_rgb, frame_resized, cv::Size(), 0.25, 0.25);
//  cv::imshow("reprojection", frame_resized);
//  cv::waitKey(-1);
  // rough calibration
  Calibration6DoF translation = Calibration::findTranslation(centers2D, centers3D, projection_matrix, radius2D,
                                                             radius3D);
//  translation.DoF[0] = tvec.at<double>(0);
//  translation.DoF[1] = tvec.at<double>(1);
//  translation.DoF[2] = tvec.at<double>(2);
//  translation.DoF[3] = euler_angles(2);
//  translation.DoF[4] = euler_angles(1);
//  translation.DoF[5] = euler_angles(0);

  if (doRefinement)
  {
    ROS_INFO("Coarse calibration:");
    translation.print();
    ROS_INFO("Refinement process started - this may take a minute.");
    size_t divisions = 5;
    float distance_transl = 0.02;
    float distance_rot = 0.01;
    Calibration6DoF best_calibration, avg_calibration;
    Calibration::calibrationRefinement(Image::Image(frame_gray),
                                       pointcloud,
                                       projection_matrix,
                                       translation.DoF[0],
                                       translation.DoF[1],
                                       translation.DoF[2],
                                       translation.DoF[3],
                                       translation.DoF[4],
                                       translation.DoF[5],
                                       distance_transl,
                                       distance_rot,
                                       divisions,
                                       best_calibration,
                                       avg_calibration);
    return avg_calibration;
  }
  else
  {
    return translation;
  }
}

void callback(const sensor_msgs::ImageConstPtr& msg_img, const sensor_msgs::CameraInfoConstPtr& msg_info,
              const sensor_msgs::PointCloud2ConstPtr& msg_pc)
{

  ROS_INFO_STREAM("Image received at " << msg_img->header.stamp.toSec());
  ROS_INFO_STREAM( "Camera info received at " << msg_info->header.stamp.toSec());
  ROS_INFO_STREAM( "Velodyne scan received at " << msg_pc->header.stamp.toSec());

  cameraMatrix.at<double>(0, 0) = msg_info->K[0];
  cameraMatrix.at<double>(0, 1) = msg_info->K[1];
  cameraMatrix.at<double>(0, 2) = msg_info->K[2];

  cameraMatrix.at<double>(1, 0) = msg_info->K[3];
  cameraMatrix.at<double>(1, 1) = msg_info->K[4];
  cameraMatrix.at<double>(1, 2) = msg_info->K[5];

  cameraMatrix.at<double>(2, 0) = msg_info->K[6];
  cameraMatrix.at<double>(2, 1) = msg_info->K[7];
  cameraMatrix.at<double>(2, 2) = msg_info->K[8];

  distCoeffs.at<double>(0) = msg_info->D[0];
  distCoeffs.at<double>(1) = msg_info->D[1];
  distCoeffs.at<double>(2) = msg_info->D[2];
  distCoeffs.at<double>(3) = msg_info->D[3];

  // Loading camera image:
  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg_img, sensor_msgs::image_encodings::BGR8);
  frame_rgb = cv_ptr->image;

  // Loading projection matrix:
  float p[12];
  float *pp = p;
  for (boost::array<double, 12ul>::const_iterator i = msg_info->P.begin(); i != msg_info->P.end(); i++)
  {
    *pp = (float)(*i);
    pp++;
  }
  cv::Mat(3, 4, CV_32FC1, &p).copyTo(projection_matrix);

  // Loading Velodyne point cloud
  PointCloud<Velodyne::Point> pc;
  fromROSMsg(*msg_pc, pc);

  // x := x, y := -z, z := y,
  pointcloud = Velodyne::Velodyne(pc).transform(0, 0, 0, 0, -M_PI/2, M_PI/2);

  // calibration:
  writeAllInputs();
  Calibration6DoF calibrationParams = calibration(doRefinement);
  if (calibrationParams.isGood())
  {
    ROS_INFO_STREAM("Calibration succeeded, found parameters:");
    calibrationParams.print();
    shutdown();
  }
  else
  {
    ROS_WARN("Calibration failed - trying again after 5s ...");
    ros::Duration(5).sleep();
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "calibration_node");

  int c;
  while ((c = getopt(argc, argv, "r")) != -1)
  {
    switch (c)
    {
      case 'r':
        doRefinement = true;
        break;
      default:
        return EXIT_FAILURE;
    }
  }

  ros::NodeHandle n;
  n.getParam("/but_calibration_camera_velodyne/camera_frame_topic", CAMERA_FRAME_TOPIC);
  n.getParam("/but_calibration_camera_velodyne/camera_info_topic", CAMERA_INFO_TOPIC);
  n.getParam("/but_calibration_camera_velodyne/velodyne_topic", VELODYNE_TOPIC);
  n.getParam("/but_calibration_camera_velodyne/marker/circles_distance", STRAIGHT_DISTANCE);
  n.getParam("/but_calibration_camera_velodyne/marker/circles_radius", RADIUS);

  message_filters::Subscriber<sensor_msgs::Image> image_sub(n, CAMERA_FRAME_TOPIC, 1);
  message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub(n, CAMERA_INFO_TOPIC, 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(n, VELODYNE_TOPIC, 1);

  typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::PointCloud2> MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, info_sub, cloud_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3));

  ros::spin();

  return EXIT_SUCCESS;
}
