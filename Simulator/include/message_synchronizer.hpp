#include <ros/ros.h>
#include <racecar_simulator/SynchronizedSensor.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vector>
class MsgSync {
public:
    MsgSync(ros::NodeHandle &nh);
    virtual ~MsgSync();
private:
    ros::NodeHandle nh_;
    ros::Subscriber scan_sub;
    ros::Subscriber odom_sub;
    ros::Subscriber collision_sub;
    image_transport::ImageTransport it;
    image_transport::Subscriber img_sub;
    ros::Publisher mega_pub;
    bool scan_here, odom_here, collision_here, image_here;
    sensor_msgs::LaserScan current_scan;
    nav_msgs::Odometry current_odom;
    std_msgs::Bool current_collision;
    sensor_msgs::Image current_image;
    void check_sync();
    void scan_callback(const sensor_msgs::LaserScan::ConstPtr &scan_msg);
    void odom_callback(const nav_msgs::Odometry::ConstPtr &odom_msg);
    void collision_callback(const std_msgs::Bool::ConstPtr &collision_msg);
    void img_callback(const sensor_msgs::Image::ConstPtr &img_msg);
};
