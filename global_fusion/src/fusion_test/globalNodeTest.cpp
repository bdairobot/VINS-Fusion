/*******************************************************
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Dai Bo (bdairobot@gmail.com)
 *******************************************************/

#include "ros/ros.h"
#include <ros/console.h>
#include "globalOptTest.h"
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/FluidPressure.h>
#include <sensor_msgs/MagneticField.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Header.h>
#include <std_msgs/Bool.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PointStamped.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <stdio.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <fstream>
#include <queue>
#include <mutex>
#include "LocalCartesian.hpp"
#include "utility/utility.h"
using namespace std;

GlobalOptimization globalEstimator;
ros::Publisher pub_global_pose, pub_gps_pose;
ros::Publisher pub_global_kf_path, pub_global_path, pub_gps_path;
ros::Publisher pub_baro_height;
ros::Publisher pub_vins_restart;
ros::Subscriber sub_myeye_imu;
nav_msgs::Path *global_kf_path, global_path, gps_path;
mutex m_buf, map_buf;

// gps information: time, x, y, z, xy_var, z_var
queue<pair<double, vector<double>>> tmpGPSQueue;
GeographicLib::LocalCartesian geoConverter;
// baro information: time, z, z_var
queue<pair<double, vector<double>>> tmpBaroQueue;
queue<pair<double, vector<double>>> tmpMagQueue;
queue<pair<double, vector<double>>> tmpVIOQueue;
queue<pair<double, vector<double>>> tmpAttQueue;

// flag bit 0: gps, 1: baro, 2: mag， 3: vio
int init_flag = 0;
int error_flag = 0;

static const float BETA_TABLE[5] = {0,
				    2.71,
				    4.61,
				    6.25,
                    7.78
				   };
static double NOISE_BARO = 1.2;
static double NOISE_MAG = 0.02;
static double NOISE_ATT = 0.05;

static uint BUF_DURATION = 3;

// aligned GPS, Baro and Mag data with VIO using time stamp
static map<double, vector<double>> map_GPS; // x,y,z,xy_var,z_var. position
static map<double, vector<double>> map_Baro; // z, z_var. position
static map<double, vector<double>> map_Mag; // x,y,z, var. nomalized mag
static map<double, vector<double>> map_Att; // w, x, y, z. var
static double myeye_t = -1.0;
static double gap_t = -1.0;

class DataAnalyses {
public:
    DataAnalyses(uint data_len):_data_len(data_len), _sum(0.0), _sum_var(0.0){}
    ~DataAnalyses(){}
    void put_data(double data){
        if (_buf_data.size() == _data_len){
            _sum -= _buf_data.front();
            _sum_var -= _buf_data.front()*_buf_data.front();
            _buf_data.pop();
        }
        _buf_data.push(data);
        _sum += data;
        _sum_var += data*data;
    }
    pair<double,double> result() {
        if (_buf_data.size() < _data_len)
            return make_pair(0.0, -1.0); // return error value
        double mean = _sum / _data_len;
        double dev = sqrt(_sum_var/_data_len - mean*mean);
        return make_pair(mean, dev);
    }
private:
    queue<double> _buf_data;
    uint _data_len;
    double _sum;
    double _sum_var;
};

void GPS_callback(const sensor_msgs::NavSatFixConstPtr &GPS_msg)
{
    if (gap_t < 0.0) return;
    double gps_t = GPS_msg->header.stamp.toSec();
    if (GPS_msg->status.status == -1){
        error_flag |= 1<<0;
        return;
    }
    static int count = 0;
    if (!(init_flag & 1<<0)){
        // position accuracy less than 2 meters or after 2 seconds
        if (GPS_msg->position_covariance[0] < 4.0 || count == 5){
            geoConverter.Reset(GPS_msg->latitude, GPS_msg->longitude, GPS_msg->altitude);
            init_flag |= 1<<0;
            ROS_INFO("geo init: lat %12.8f,lon %12.8f, alt %12.8f, cov: %8.4f", GPS_msg->latitude,GPS_msg->longitude, GPS_msg->altitude, GPS_msg->position_covariance[0]);
        }
        count++;
    }
    if (init_flag & 1<<0) {
        
        geometry_msgs::PoseStamped pose_gps;
        pose_gps.header.stamp = ros::Time(gps_t);
        pose_gps.header.frame_id = "world";
        double xyz[3];
        geoConverter.Forward(GPS_msg->latitude, GPS_msg->longitude,GPS_msg->altitude, xyz[0],xyz[1],xyz[2]);
           //printf("gps_callback! ");
        vector<double> gps_info = {xyz[0],xyz[1],xyz[2], GPS_msg->position_covariance[0], GPS_msg->position_covariance[8]};
        m_buf.lock();
        tmpGPSQueue.push(make_pair(gps_t, gps_info));
        // store 3s data
        m_buf.unlock();
        if (tmpGPSQueue.size() > 5 * BUF_DURATION)
            tmpGPSQueue.pop();
        pose_gps.pose.position.x = xyz[0];
        pose_gps.pose.position.y = xyz[1];
        pose_gps.pose.position.z = xyz[2];
        pose_gps.pose.orientation.w = 1.0;
        pub_gps_pose.publish(pose_gps);
        gps_path.header = pose_gps.header;
        gps_path.poses.push_back(pose_gps);
        pub_gps_path.publish(gps_path);
    }
}

void baro_callback(const sensor_msgs::FluidPressureConstPtr &baro_msg)
{   
    if(gap_t < 0) return;
    double baro_t = baro_msg->header.stamp.toSec();
    static double CONSTANTS_ABSOLUTE_NULL_CELSIUS = -273.15;
    static double CONSTANTS_AIR_GAS_CONST = 287.1f;
    static double BARO_MSL = 101.325; /* current pressure at MSL in kPa */
    static double CONSTANTS_ONE_G = 9.80665f;
    static double T1 = 15.0 - CONSTANTS_ABSOLUTE_NULL_CELSIUS;	/* temperature at base height in Kelvin */
    static double a  = -6.5 / 1000.0;	/* temperature gradient in degrees per metre */

    /* measured pressure in kPa */
    const double p = baro_msg->fluid_pressure * 0.001;
    /*
        * Solve:
        *
        *     /               -(aR / g)    \
        *    | (p / BARO_MSL)          . T1 | - T1
        *     \                            /
        * h = -------------------------------  + h1
        *                   a
        */
    double height = (((pow((p / BARO_MSL), (-(a * CONSTANTS_AIR_GAS_CONST) / CONSTANTS_ONE_G))) * T1) - T1) / a;
    static double init_height = 0.0;
    static double last_height = height;
    double filter_factor = 0.5;
    double filter_height = filter_factor*height + (1-filter_factor) * last_height;
    last_height = filter_height;
    m_buf.lock();
    vector<double> baro_info = {filter_height - init_height, NOISE_BARO*NOISE_BARO};
    tmpBaroQueue.push(make_pair(baro_t,baro_info));
    m_buf.unlock();

    // store 3s data
    if (tmpBaroQueue.size() > 50*BUF_DURATION)
        tmpBaroQueue.pop();
    static unsigned int init_size = 20;
    if (!(init_flag & 1<<1) && tmpBaroQueue.size() == init_size){        
        init_height = filter_height;
        while (!tmpBaroQueue.empty())
            tmpBaroQueue.pop();
        init_flag |= 1<<1;
        ROS_INFO("init baro: height %8.4f", init_height);
    }
    if (init_flag & 1<<1){
        geometry_msgs::PointStamped baro_height;
        baro_height.header = baro_msg->header;
        baro_height.point.x = height - init_height;
        baro_height.point.y = filter_height - init_height;
        pub_baro_height.publish(baro_height);

        // static DataAnalyses data_analy(100);
        // data_analy.put_data(height - init_height);
        // if (data_analy.result().second > 0.0)
        //     ROS_INFO("baro mean: %8.4f, dev: %8.4f", data_analy.result().first, data_analy.result().second);
    }
}

void mag_callback(const sensor_msgs::MagneticFieldConstPtr &mag_msg)
{   
    if (gap_t < 0) return;
    double mag_t = mag_msg->header.stamp.toSec();
    double mag_norm = sqrt(mag_msg->magnetic_field.x*mag_msg->magnetic_field.x + mag_msg->magnetic_field.y*mag_msg->magnetic_field.y + mag_msg->magnetic_field.z*mag_msg->magnetic_field.z);
    double mag_x = mag_msg->magnetic_field.x / mag_norm;
    double mag_y = mag_msg->magnetic_field.y / mag_norm;
    double mag_z = mag_msg->magnetic_field.z / mag_norm;
    m_buf.lock();
    tmpMagQueue.push(make_pair(mag_t, vector<double>{mag_x, mag_y, mag_z, NOISE_MAG*NOISE_MAG}));
    m_buf.unlock();
    if (tmpMagQueue.size() > 50*BUF_DURATION)
        tmpMagQueue.pop();

    // static DataAnalyses data_analy_x(100);
    // static DataAnalyses data_analy_y(100);
    // static DataAnalyses data_analy_z(100);
    // data_analy_x.put_data(mag_x);
    // data_analy_y.put_data(mag_y);
    // data_analy_z.put_data(mag_z);
    // if (data_analy_x.result().second > 0.0)
    //     ROS_INFO("mean: %8.4f, %8.4f, %8.4f", data_analy_x.result().first, data_analy_y.result().first, data_analy_z.result().first);
}

void keyframe_callback(const nav_msgs::OdometryConstPtr &pose_msg)
{
    if (gap_t < 0) return;
    if (!(error_flag & 1<<3 && pose_msg->pose.covariance[0]>=0.0)){
        double kf_t = pose_msg->header.stamp.toSec() + gap_t;
        Eigen::Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
        Eigen::Quaterniond vio_q;
        vio_q.w() = pose_msg->pose.pose.orientation.w;
        vio_q.x() = pose_msg->pose.pose.orientation.x;
        vio_q.y() = pose_msg->pose.pose.orientation.y;
        vio_q.z() = pose_msg->pose.pose.orientation.z;

        map_buf.lock();
        if(map_GPS.find(kf_t) != map_GPS.end() && !(error_flag & 1<<0))
            globalEstimator.inputGPS(kf_t, map_GPS[kf_t]);
        if(map_Baro.find(kf_t) != map_Baro.end())
            globalEstimator.inputBaro(kf_t, map_Baro[kf_t]);
        if(map_Att.find(kf_t) != map_Att.end())
            globalEstimator.inputAtt(kf_t, map_Att[kf_t]);
        if(map_Mag.find(kf_t) != map_Mag.end()){
            // if (!globalEstimator.mag_init){
            //     Eigen::Vector3d ypr = Utility::R2ypr(vio_q.toRotationMatrix());
            //     Eigen::Matrix3d R = Utility::ypr2R(Eigen::Vector3d{0.0, ypr(1), ypr(2)});
            //     Eigen::Vector3d mag = R*Eigen::Vector3d(map_Mag[kf_t][0],map_Mag[kf_t][1],map_Mag[kf_t][2]);
            //     double yaw_mag = atan2(mag(0),mag(1)) * 180/3.14159; // East is zero degree
            //     std::cout<< "mag yaw: " << yaw_mag << std::endl;
            //     globalEstimator.WGPS_T_WVIO.block<3,3>(0,0) = Utility::ypr2R(Eigen::Vector3d{yaw_mag, ypr(1), ypr(2)})*vio_q.inverse().toRotationMatrix();

            //     globalEstimator.mag_init = true;
            // }
            globalEstimator.inputMag(kf_t, map_Mag[kf_t]);
        }
        globalEstimator.inputKeyframe(kf_t, vio_t, vio_q, pose_msg->pose.covariance[0], pose_msg->pose.covariance[28]);
        pub_global_kf_path.publish(*global_kf_path);
        map_buf.unlock();
    }
}

void vio_callback(const nav_msgs::OdometryConstPtr &pose_msg)
{
    if (gap_t < 0) return;
    //printf("vio_callback! ");
    if(pose_msg->pose.covariance[0] < 0){
        error_flag |= 1<<3;
        init_flag &= 0<<3;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_vins_restart.publish(restart_flag);
        return;
    }
    if (!(init_flag & 1<<3)){
        init_flag |= 1<<3;
    }
    error_flag &= 0<<3;
    double t = pose_msg->header.stamp.toSec() + gap_t;
    Eigen::Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
    Eigen::Quaterniond vio_q;
    vio_q.w() = pose_msg->pose.pose.orientation.w;
    vio_q.x() = pose_msg->pose.pose.orientation.x;
    vio_q.y() = pose_msg->pose.pose.orientation.y;
    vio_q.z() = pose_msg->pose.pose.orientation.z;
    Eigen::Vector3d global_t;
    Eigen::Quaterniond global_q;
    globalEstimator.getGlobalOdom(vio_t, vio_q, global_t, global_q);
    geometry_msgs::PoseStamped vio_pose;
    vio_pose.header = pose_msg->header;
    vio_pose.header.frame_id = "world";
    vio_pose.pose.position.x = global_t.x();
    vio_pose.pose.position.y = global_t.y();
    vio_pose.pose.position.z = global_t.z();
    vio_pose.pose.orientation.x = global_q.x();
    vio_pose.pose.orientation.y = global_q.y();
    vio_pose.pose.orientation.z = global_q.z();
    vio_pose.pose.orientation.w = global_q.w();
    pub_global_pose.publish(vio_pose);
    global_path.poses.push_back(vio_pose);
    pub_global_path.publish(global_path);

    static uint duration = 4;
    { // align with baro
        static uint step = 10;
        if (init_flag & 1<<1){
            while(!tmpBaroQueue.empty()){
                double baro_t = tmpBaroQueue.front().first;
                if (t < baro_t - 0.02 && (tmpBaroQueue.size() == 50*BUF_DURATION)){
                    ROS_ERROR("VIO is away behind BARO information! ");
                    ROS_INFO("t:%8.4f baro_t: %8.4f", t, baro_t);
                    assert(0);
                } else if (t <= baro_t + 0.02 && t >= baro_t - 0.02){
                    map_buf.lock();
                    if (map_Baro.size() == duration*step){
                        static bool shown_once = false;
                        if(!shown_once) {ROS_INFO("map_Baro.size(): %lu ", map_Baro.size()); shown_once = true;}
                        map_Baro.erase(map_Baro.begin());
                    }
                    map_Baro[t] = tmpBaroQueue.front().second;
                    map_buf.unlock();
                    tmpBaroQueue.pop();
                    break;
                } else
                    tmpBaroQueue.pop();
            }
        }
    }

    { // align with mag
        static uint step = 10;
        while(!tmpMagQueue.empty()){
            double mag_t = tmpMagQueue.front().first;
            if (t < mag_t - 0.02 && (tmpMagQueue.size() == 50*BUF_DURATION)){
                ROS_ERROR("VIO is away behind MAG information! ");
                assert(0);
            } else if (t <= mag_t + 0.02 && t >= mag_t - 0.02){
                map_buf.lock();
                if (map_Mag.size() == duration*step){
                    static bool shown_once = false;
                    if(!shown_once) {ROS_INFO("map_Mag.size(): %lu ", map_Mag.size()); shown_once = true;}
                    map_Mag.erase(map_Mag.begin());
                }
                map_Mag[t] = tmpMagQueue.front().second;
                map_buf.unlock();
                tmpMagQueue.pop();
                break;
            } else
                tmpMagQueue.pop();
        }
    }

    { // align with att
        static uint step = 10;
        while(!tmpAttQueue.empty()){
            double att_t = tmpAttQueue.front().first;
            if (t < att_t - 0.02 && (tmpAttQueue.size() == 50*BUF_DURATION)){
                ROS_ERROR("VIO is away behind MAG information! ");
                assert(0);
            } else if (t <= att_t + 0.02 && t >= att_t - 0.02){
                map_buf.lock();
                if (map_Att.size() == duration*step){
                    static bool shown_once = false;
                    if(!shown_once) {ROS_INFO("map_Att.size(): %lu ", map_Att.size()); shown_once = true;}
                    map_Att.erase(map_Att.begin());
                }
                map_Att[t] = tmpAttQueue.front().second;
                map_buf.unlock();
                tmpAttQueue.pop();
                break;
            } else
                tmpAttQueue.pop();
        }
    }

    { // check with gps
        static uint step = 5;
        if (init_flag & 1<<0){
            while(!tmpGPSQueue.empty()){
                pair<double, vector<double>> GPS_info = tmpGPSQueue.front();
                double gps_t = GPS_info.first;
                Eigen::Vector3d gps_pose(GPS_info.second[0],GPS_info.second[1],GPS_info.second[2]);
                static pair<double, vector<double>> last_pop;
                static bool pop_flag = false;
                static bool shown_once = false;
                if (t < gps_t - 0.02 && (tmpGPSQueue.size()==15)){
                    ROS_ERROR("VIO is away behind GPS information! ");
                    assert(0);
                } else if (t <= gps_t + 0.02 && t >= gps_t - 0.02){
                    map_buf.lock();
                    if (map_GPS.size() == duration*step){
                        if(!shown_once) {ROS_INFO("map_GPS.size(): %lu ", map_GPS.size()); shown_once = true;}
                        map_GPS.erase(map_GPS.begin());
                    }
                    map_GPS[t] = GPS_info.second;
                    last_pop = GPS_info;
                    tmpGPSQueue.pop();
                    pop_flag = true;
                    map_buf.unlock();
                    break;
                }else { // insert
                    if(!pop_flag || t > gps_t + 0.02){
                        last_pop = GPS_info;
                        tmpGPSQueue.pop();
                        pop_flag = true;
                    } else {
                        Eigen::Vector3d insert_gps_pose = gps_pose - (gps_t - t)/(gps_t - last_pop.first)*(gps_pose - Eigen::Vector3d{last_pop.second[0],last_pop.second[1],last_pop.second[2]});
                        map_buf.lock();
                        if (map_GPS.size() == duration*step){
                            map_GPS.erase(map_GPS.begin());
                            if(!shown_once) {ROS_INFO("map_GPS.size(): %lu ", map_GPS.size()); shown_once = true;}
                        }
                        map_GPS[t] = vector<double> {insert_gps_pose(0), insert_gps_pose(1), insert_gps_pose(2), GPS_info.second[3], GPS_info.second[4]};
                        last_pop = GPS_info;
                        tmpGPSQueue.pop();
                        pop_flag = true;
                        map_buf.unlock();
                        break;
                    }
                }
            }
        }
    }
}

void myeye_imu_callback(const sensor_msgs::ImuConstPtr &imu_msg){
    myeye_t = imu_msg->header.stamp.toSec();
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg){
    double att_t = imu_msg->header.stamp.toSec();
    if (myeye_t >= 0.0 && gap_t < 0.0){
        if (att_t > myeye_t + 3.0)
            gap_t = att_t - myeye_t;
        else gap_t = 0.0;
        ROS_INFO("gap_t: %f", gap_t);
        sub_myeye_imu.shutdown();
    }
    m_buf.lock();
    vector<double> att_info = {imu_msg->orientation.w, imu_msg->orientation.x, imu_msg->orientation.y, imu_msg->orientation.z, NOISE_ATT*NOISE_ATT};
    tmpAttQueue.push(make_pair(att_t, att_info));
    m_buf.unlock();

    if(tmpAttQueue.size() > 50*BUF_DURATION)
        tmpAttQueue.pop();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "globalEstimator");
    ros::NodeHandle n("~");
    global_kf_path = &globalEstimator.global_path;

    ros::Subscriber sub_GPS = n.subscribe("/mavros/global_position/raw/fix", 100, GPS_callback);
    ros::Subscriber sub_baro = n.subscribe("/mavros/imu/static_pressure",100, baro_callback);
    // ros::Subscriber sub_mag = n.subscribe("/mavros/imu/mag",100, mag_callback);
    ros::Subscriber sub_kf = n.subscribe("/vins_estimator/keyframe_pose", 100, keyframe_callback);
    ros::Subscriber sub_vio = n.subscribe("/vins_estimator/odometry", 100, vio_callback);
    sub_myeye_imu = n.subscribe("/mynteye/imu/data_raw", 100, myeye_imu_callback);
    ros::Subscriber sub_imu = n.subscribe("/mavros/imu/data", 100, imu_callback);

    pub_global_path = n.advertise<nav_msgs::Path>("global_path", 100);
    pub_global_kf_path = n.advertise<nav_msgs::Path>("global_kf_path", 100);
    pub_gps_path = n.advertise<nav_msgs::Path>("gps_path", 100);
    pub_global_pose = n.advertise<geometry_msgs::PoseStamped>("global_pose", 100);
    pub_gps_pose = n.advertise<geometry_msgs::PoseStamped>("gps_pose", 100);
    pub_baro_height = n.advertise<geometry_msgs::PointStamped>("baro_height", 100);
    pub_vins_restart = n.advertise<std_msgs::Bool>("/vins_restart", 5);

    ros::spin();
    return 0;
}
