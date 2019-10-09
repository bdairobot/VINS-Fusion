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
static Eigen::Vector3d mag_w(0.0000,0.54159435858580351,-0.84063996);
struct magFactorAuto
{
	magFactorAuto(double x, double y, double z, double dev)
				  :x(x), y(y), z(z), dev(dev){}

	template <typename T>
	bool operator()(const T* const q_w_i, T* residuals) const
	{
        	T q_inv[4] = {T(q_w_i[0]), T(-q_w_i[1]), T(-q_w_i[2]), T(-q_w_i[3])};
		T mag_w_tmp[3] = {T(mag_w(0)), T(mag_w(1)), T(mag_w(2))};
		T mag[3];
		ceres::QuaternionRotatePoint(q_inv, mag_w_tmp, mag);

		residuals[0] =(T(x)-mag[0]) / T(dev);
		residuals[1] =(T(y)-mag[1]) / T(dev);
		residuals[2] =(T(z)-mag[2]) / T(dev);

		return true;
	}

	static ceres::CostFunction* Create(const double x, const double y, const double z, 
                                    const double dev) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          magFactorAuto, 3, 4>(
	          	new magFactorAuto(x, y, z, dev)));
	}

	double x, y, z;
	double dev;

};
