/*******************************************************
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Dai Bo (bdairobot@gmail.com)
 *******************************************************/

#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>

struct simFactorAuto
{
	simFactorAuto(Eigen::Quaterniond q_w, Eigen::Vector3d p_w, Eigen::Quaterniond q_v, Eigen::Vector3d p_v, double sim_scale):
		q_w(q_w), p_w(q_w), q_v(q_v), p_v(p_v), sim_scale(sim_scale) {}

	template <typename T>
	bool operator() (const T* const q_w_v, const T* const t_w_v, T* residuals) const
	{
		T p_v_k[3] = {T(p_v.x()), T(p_v.y()), T(p_v.z())};
		T q_v_k[4] = {T(q_v.w()), T(q_v.x()), T(q_v.y()), T(q_v.z())};
		T p_w_k[3] = {T(p_w.x()), T(p_w.y()), T(p_w.z())};
		T q_w_k[4] = {T(q_w.w()), T(q_w.x()), T(q_w.y()), T(q_w.z())};

		T tmp_p[3];
		ceres::QuaternionRotatePoint(q_w_v, p_v_k, tmp_p);
		residuals[0] = (p_w_k[0] - (T(sim_scale) * tmp_p[0] +  t_w_v[0])) / T(1);
		residuals[1] = (p_w_k[1] - (T(sim_scale) * tmp_p[1] +  t_w_v[1])) / T(1);
		residuals[2] = (p_w_k[2] - (T(sim_scale) * tmp_p[2] +  t_w_v[2])) / T(1);

		T tmp_q[4];
		ceres::QuaternionProduct(q_w_v, q_v_k, tmp_q);
		T q_w_inv = {T(q_w[0]), -T(q_w[1]), -T(q_w[2]), -T(q_w[3])};
		T error_q[4];
		ceres::QuaternionProduct(q_w_inv, tmp_q, error_q);
		residuals[3] = T(2) * error_q[1] / T(1);
		residuals[4] = T(2) * error_q[2] / T(1);
		residuals[5] = T(2) * error_q[3] / T(1);

		return true;
	}

	static ceres::CostFunction* Create(const Eigen::Quaterniond q_w, const Eigen::Vector3d p_w, const Eigen::Quaterniond q_v, const Eigen::Vector3d p_v, const double sim_scale)
	{
		return (new ceres::AutoDiffCostFunction<simFactorAuto, 6, 4, 3> (
			new simFactorAuto(q_w, p_w, q_v, p_v)));
	}

	Eigen::Quaterniond q_w, q_v;
	Eigen::Vector3d p_w, p_v;
	double sim_scale;
};