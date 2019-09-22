/*******************************************************
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Dai Bo (bdairobot@gmail.com)
 *******************************************************/

#include "globalOptTest.h"
#include "factors/baroFactor.h"
#include "factors/GPSFactor.h"
// #include "factors/VIOFactor.h"
// #include "factors/attFactor.h"
#include "factors/attFactorAuto.h"
#include "factors/VIOFactorAuto.h"
#include "factors/localAlignAuto.h"
#include <fstream>

GlobalOptimization::GlobalOptimization()
{
    newGPS = false;
    newBaro = false;
    newAtt = false;
    att_init = false;
    pos_init = false;
    mag_decl[0][0] = -0.0 / 180.0 * 3.14159;
	WGPS_T_WVIO = Eigen::Matrix4d::Identity();
    sim_scale[0] = 1.0;
    threadOpt = std::thread(&GlobalOptimization::optimize, this);
}

GlobalOptimization::~GlobalOptimization()
{
    threadOpt.detach();
}

void GlobalOptimization::restart()
{
    mPoseMap.lock();
    newGPS = newBaro = newAtt =false;
    att_init = pos_init = false;
    WGPS_T_WVIO = Eigen::Matrix4d::Identity();
    localPoseMap.clear();
    globalPoseMap.clear();
    GPSPositionMap.clear();
    baroMap.clear();
    attMap.clear();
    mPoseMap.unlock();

}

void GlobalOptimization::inputKeyframe(double t, Eigen::Vector3d OdomP, Eigen::Quaterniond OdomQ, double p_var, double q_var)
{
    mPoseMap.lock();
    vector<double> localPose{OdomP.x(), OdomP.y(), OdomP.z(), 
    					     OdomQ.w(), OdomQ.x(), OdomQ.y(), OdomQ.z(), p_var, q_var};
    localPoseMap[t] = localPose;
    Eigen::Quaterniond globalQ;
    globalQ = WGPS_T_WVIO.block<3, 3>(0, 0) * OdomQ;
    Eigen::Vector3d globalP = sim_scale[0]*WGPS_T_WVIO.block<3, 3>(0, 0) * OdomP + WGPS_T_WVIO.block<3, 1>(0, 3);
    vector<double> globalPose{globalP.x(), globalP.y(), globalP.z(),
                              globalQ.w(), globalQ.x(), globalQ.y(), globalQ.z()};
    globalPoseMap[t] = globalPose;
    mPoseMap.unlock();

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time(t);
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = globalP.x();
    pose_stamped.pose.position.y = globalP.y();
    pose_stamped.pose.position.z = globalP.z();
    pose_stamped.pose.orientation.x = globalQ.x();
    pose_stamped.pose.orientation.y = globalQ.y();
    pose_stamped.pose.orientation.z = globalQ.z();
    pose_stamped.pose.orientation.w = globalQ.w();
    global_path.header = pose_stamped.header;
    global_path.poses.push_back(pose_stamped);
}

void GlobalOptimization::getGlobalOdom(Eigen::Vector3d OdomP, Eigen::Quaterniond OdomQ, Eigen::Vector3d &global_odomP, Eigen::Quaterniond &global_odomQ)
{
    global_odomQ = WGPS_T_WVIO.block<3, 3>(0, 0) * OdomQ;
    global_odomP = sim_scale[0] * WGPS_T_WVIO.block<3, 3>(0, 0) * OdomP + WGPS_T_WVIO.block<3, 1>(0, 3);
}

void GlobalOptimization::inputGPS(double gps_t, vector<double> &GPS_pose)
{
    //printf("new gps: t: %f x: %f y: %f z:%f \n", t, tmp[0], tmp[1], tmp[2]);

    GPSPositionMap.insert(make_pair(gps_t, GPS_pose));
    newGPS = true;
}

void GlobalOptimization::inputBaro(double baro_t, vector<double> &Baro)
{
    baroMap.insert(make_pair(baro_t, Baro));
    newBaro = true;
}
void GlobalOptimization::inputAtt(double att_t, vector<double> &Att)
{
    attMap.insert(make_pair(att_t, Att));
    newAtt = true;
}

void GlobalOptimization::optimize()
{
    std::ofstream fout_time("/home/bdai/output/global_opt_time.txt", std::ios::out);
    fout_time.close();
    static int min_size = 10, max_local_size =80;
    while(true){
        if(newGPS || newAtt || newBaro){
            int length = localPoseMap.size();
            if (length < min_size) {
                std::chrono::milliseconds dura(500);
                std::this_thread::sleep_for(dura);
                std::cout << "No optimization, localPoseMap size is: " << length << ", it must > " << min_size << std::endl;
                continue;
            }
            // std::cout << "global optimization!" << std::endl;
            TicToc globalOptimizationTime;
            ceres::Problem problem;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            //options.minimizer_progress_to_stdout = true;
            //options.max_solver_time_in_seconds = SOLVER_TIME * 3;
            options.max_num_iterations = 5;
            ceres::Solver::Summary summary;
            ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
            ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();

            //add global parameters
            mPoseMap.lock();
            
            // w^t_i   w^q_i
            double t_array_xy[length][2];
            double t_array_z[length][1];
            double q_array[length][4];
            map<double, vector<double>>::iterator iter;
            iter = globalPoseMap.begin();
            for (int i = 0; i < length; i++, iter++){
                t_array_xy[i][0] = iter->second[0];
                t_array_xy[i][1] = iter->second[1];
                t_array_z[i][0] = iter->second[2];
                q_array[i][0] = iter->second[3];
                q_array[i][1] = iter->second[4];
                q_array[i][2] = iter->second[5];
                q_array[i][3] = iter->second[6];
                problem.AddParameterBlock(q_array[i], 4, local_parameterization);
                problem.AddParameterBlock(t_array_xy[i], 2);
                problem.AddParameterBlock(t_array_z[i], 1);
                if (i >= length -2)
                    std::cout << "Address: " << std::hex << q_array[i] << ", " << t_array_xy[i] << ", " << t_array_z[i] << std::dec << std::endl;
            }
            // add mag decline degree
            problem.AddParameterBlock(mag_decl[0], 1);
            problem.AddParameterBlock(sim_scale, 1);
            //if (!newGPS)
            //    problem.SetParameterBlockConstant(mag_decl[0]);
            static int opt_length = 2000;
            if (length > opt_length){
                for (int i = 0; i < length - opt_length; i++) {
                    problem.SetParameterBlockConstant(q_array[i]);
                    problem.SetParameterBlockConstant(t_array_xy[i]);
                    problem.SetParameterBlockConstant(t_array_z[i]);
                }
            }
            std::cout << "mag_decl: " << mag_decl[0][0] * 180.0/3.14159 << " VIO size: " << localPoseMap.size() << " GPS size: " << GPSPositionMap.size() << " ATT size: " << attMap.size() << " Baro size: " << baroMap.size() << std::endl;

            map<double, vector<double>>::iterator iterVIO, iterVIONext, iterGPS, iterBaro, iterAtt;
            int i = 0;
            bool update_scale = true;
            for (iterVIO = localPoseMap.begin(); iterVIO != localPoseMap.end(); iterVIO++, i++){
                // add vio factor
                iterVIONext = iterVIO;
                iterVIONext++;
                if(iterVIONext != localPoseMap.end()){
                    // Eigen::Vector3d p_i(iterVIO->second[0], iterVIO->second[1], iterVIO->second[2]);
                    // Eigen::Quaterniond q_i(iterVIO->second[3], iterVIO->second[4], iterVIO->second[5], iterVIO->second[6]);
                    // Eigen::Vector3d p_j(iterVIONext->second[0], iterVIONext->second[1], iterVIONext->second[2]);
                    // Eigen::Quaterniond q_j(iterVIONext->second[3], iterVIONext->second[4], iterVIONext->second[5], iterVIONext->second[6]);
                   
                    // // VIOFactors* vio_cost = new VIOFactors(p_i, q_i, p_j, q_j, 1.0, 1.0);
                    // VIOFactors* vio_cost = new VIOFactors(p_i, q_i, p_j, q_j, 0.1, 0.01);
                    // problem.AddResidualBlock(vio_cost, loss_function, t_array_xy[i], t_array_z[i], q_array[i], t_array_xy[i+1], t_array_z[i+1], q_array[i+1]);

                    // double **param = new double* [6];
                    // param[0] = t_array_xy[i];
                    // param[1] = t_array_z[i];
                    // param[2] = q_array[i];
                    // param[3] = t_array_xy[i+1];
                    // param[4] = t_array_z[i+1];
                    // param[5] = q_array[i+1];
                    // vio_cost->check(param);

                    Eigen::Matrix4d wTi = Eigen::Matrix4d::Identity();
                    Eigen::Matrix4d wTj = Eigen::Matrix4d::Identity();
                    wTi.block<3, 3>(0, 0) = Eigen::Quaterniond(iterVIO->second[3], iterVIO->second[4], 
                                                               iterVIO->second[5], iterVIO->second[6]).toRotationMatrix();
                    wTi.block<3, 1>(0, 3) = Eigen::Vector3d(iterVIO->second[0], iterVIO->second[1], iterVIO->second[2]);
                    wTj.block<3, 3>(0, 0) = Eigen::Quaterniond(iterVIONext->second[3], iterVIONext->second[4], 
                                                               iterVIONext->second[5], iterVIONext->second[6]).toRotationMatrix();
                    wTj.block<3, 1>(0, 3) = Eigen::Vector3d(iterVIONext->second[0], iterVIONext->second[1], iterVIONext->second[2]);
                    Eigen::Matrix4d iTj = wTi.inverse() * wTj;
                    Eigen::Quaterniond iQj;
                    iQj = iTj.block<3, 3>(0, 0);
                    Eigen::Vector3d iPj = iTj.block<3, 1>(0, 3);

                    if (i >= length - max_local_size){ // factor for sim scale
                        ceres::CostFunction* vio_function = VIOSimFactorAuto::Create(iPj.x(), iPj.y(), iPj.z(),
                                                                                iQj.w(), iQj.x(), iQj.y(), iQj.z(),
                                                                                0.1, 0.01);
                        problem.AddResidualBlock(vio_function, nullptr, q_array[i], t_array_xy[i], t_array_z[i], q_array[i+1], t_array_xy[i+1],t_array_z[i+1], sim_scale);
                    } else {
                        ceres::CostFunction* vio_function = VIOFactorAuto::Create(iPj.x(), iPj.y(), iPj.z(),
                                                                                iQj.w(), iQj.x(), iQj.y(), iQj.z(),
                                                                                0.1, 0.01);
                        problem.AddResidualBlock(vio_function, nullptr, q_array[i], t_array_xy[i], t_array_z[i], q_array[i+1], t_array_xy[i+1],t_array_z[i+1]);
                    }
                }
                
                double t = iterVIO->first;

                /* gps factor */
                iterGPS = GPSPositionMap.find(t);
                if (iterGPS != GPSPositionMap.end()){
                    GPSFactor* gps_cost = new GPSFactor(iterGPS->second[0], iterGPS->second[1], iterGPS->second[2], sqrt(iterGPS->second[3]), 100);
                    problem.AddResidualBlock(gps_cost, loss_function, t_array_xy[i], t_array_z[i]);
                    if (i >= length - max_local_size) {
                       update_scale = (update_scale && (
                            (iterGPS->second[3] < 16 && iterGPS->second[3] >= 9 && iterGPS->second[5]<0.15) || 
                            (iterGPS->second[3] < 9 && iterGPS->second[3] >= 4 && iterGPS->second[5]<0.1) || 
                            (iterGPS->second[3] < 4  && iterGPS->second[5]<0.05)));
                    }
                    // double **param = new double* [2];
                    // param[0] = t_array_xy[i];
                    // param[1] = t_array_z[i];
                    // gps_cost->check(param);
                }
                // /* baro factor */
                iterBaro = baroMap.find(t);
                if (iterBaro != baroMap.end()){
                    BaroFactor* baro_cost = new BaroFactor(iterBaro->second[0], sqrt(iterBaro->second[1]));
                    problem.AddResidualBlock(baro_cost, loss_function, t_array_z[i]);
                }

                /* att factor */
                iterAtt = attMap.find(t);
		        // if (false) {
                if (iterAtt != attMap.end()){
                    ceres::CostFunction* att_cost = attFactorAuto::Create(iterAtt->second[0], iterAtt->second[1], iterAtt->second[2],iterAtt->second[3], sqrt(iterAtt->second[4]));
                    problem.AddResidualBlock(att_cost, loss_function, q_array[i], mag_decl[0]);
                }
            }
            mPoseMap.unlock();

            if (!update_scale || !newGPS) 
                problem.SetParameterBlockConstant(sim_scale);
            ceres::Solve(options, &problem, &summary);
            // std::cout << summary.BriefReport() << "\n";

            // update global pose
            mPoseMap.lock();
            iter = globalPoseMap.begin();
            ceres::Problem problem_l;
            ceres::Solver::Options options_l;
            options_l.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options_l.max_num_iterations = 5;
            ceres::Solver::Summary summary_l;
            ceres::LossFunction *loss_function_l = new ceres::HuberLoss(0.8);
            ceres::LocalParameterization* local_parameterization_l = new ceres::QuaternionParameterization();
            double q_relate[4];
            double t_relate[3];
            // std::cout << "q_relate address: " << std::hex << q_relate <<", " << t_relate << ", " << std::dec << std::endl;
            static int relate_opt_size = 30;

            for (int i = 0; i < length; i++, iter++)
            {
            	vector<double> globalPose{t_array_xy[i][0], t_array_xy[i][1], t_array_z[i][0],
            							  q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]};
            	iter->second = globalPose;
                // local optimization get relation between vision and global
                
            	if((length - relate_opt_size >=0) && (i >= length - relate_opt_size))
            	{   
                    double t = iter->first;
                    Eigen::Quaterniond q_v(localPoseMap[t][3], localPoseMap[t][4], localPoseMap[t][5], localPoseMap[t][6]);
                    Eigen::Vector3d p_v(localPoseMap[t][0], localPoseMap[t][1], localPoseMap[t][2]);
                    Eigen::Quaterniond q_w(globalPoseMap[t][3], globalPoseMap[t][4], globalPoseMap[t][5], globalPoseMap[t][6]);
                    Eigen::Vector3d p_w(globalPoseMap[t][0], globalPoseMap[t][1], globalPoseMap[t][2]);
                    if (i == length - relate_opt_size){
                        Eigen::Quaterniond q_w_v(WGPS_T_WVIO.block<3,3>(0,0));
                        Eigen::Vector3d t_w_v = WGPS_T_WVIO.block<3,1>(0,3);
                        q_relate[0] = q_w_v.w();
                        q_relate[1] = q_w_v.x();
                        q_relate[2] = q_w_v.y();
                        q_relate[3] = q_w_v.z();
                        t_relate[0] = t_w_v.x();
                        t_relate[1] = t_w_v.y();
                        t_relate[2] = t_w_v.z();
                        problem_l.AddParameterBlock(q_relate, 4, local_parameterization_l);
                        problem_l.AddParameterBlock(t_relate, 3);
                    }
                    // std::cout << q_relate[0] << ", "<< q_relate[1] << ", "<< q_relate[2] << ", "<< q_relate[3] << ", "<< t_relate[0] << ", "<< t_relate[1] << ", "<< t_relate[2] << std::endl;
                    // std::cout << q_w.vec().transpose() << ", " << p_w.transpose() << ", " << q_v.vec().transpose() << ", " <<  p_v.transpose() << ", " << sim_scale[0] << std::endl;
                    ceres::CostFunction* cost_fun = localAlignAuto::Create(q_w, p_w, q_v, p_v, sim_scale[0]);
                    problem_l.AddResidualBlock(cost_fun, loss_function_l, q_relate, t_relate);
            	}
            }
            if (length >= relate_opt_size){
                ceres::Solve(options_l, &problem_l, &summary_l);
                std::cout << "scale: " << sim_scale[0] << std::endl;
                Eigen::Quaterniond q_w_v(q_relate[0], q_relate[1], q_relate[2], q_relate[3]);
                Eigen::Vector3d t_w_v(t_relate[0], t_relate[1], t_relate[2]);
                WGPS_T_WVIO.block<3,3>(0,0) = q_w_v.toRotationMatrix();
                WGPS_T_WVIO.block<3,1>(0,3) = t_w_v;
            }
            newGPS = newBaro = newAtt = false;
            updateGlobalPath();
            mPoseMap.unlock();
            printf("global time %f \n", globalOptimizationTime.toc());
            ofstream fout_time("/home/bdai/output/global_opt_time.txt", std::ios::app);
            fout_time.setf(ios::fixed, ios::floatfield);
            fout_time.precision(6);
            fout_time << localPoseMap.rbegin()->first << " " << globalOptimizationTime.toc() << " " << sim_scale[0] << endl;
            fout_time.close();
        }
        std::chrono::milliseconds dura(50);
        std::this_thread::sleep_for(dura);
        // std::cout << "GPSPositionMap size: " << GPSPositionMap.size() << std::endl;

    }
	return;
}


void GlobalOptimization::updateGlobalPath()
{
    global_path.poses.clear();
    map<double, vector<double>>::iterator iter;
    for (iter = globalPoseMap.begin(); iter != globalPoseMap.end(); iter++)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time(iter->first);
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose.position.x = iter->second[0];
        pose_stamped.pose.position.y = iter->second[1];
        pose_stamped.pose.position.z = iter->second[2];
        pose_stamped.pose.orientation.w = iter->second[3];
        pose_stamped.pose.orientation.x = iter->second[4];
        pose_stamped.pose.orientation.y = iter->second[5];
        pose_stamped.pose.orientation.z = iter->second[6];
        global_path.poses.push_back(pose_stamped);
    }
}
