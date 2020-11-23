#ifndef __EVL_SE3__
#define __EVL_SE3__

#include <opencv2/core.hpp>

namespace evl
{
    /**
    * \brief 3차원 Special Euclidean Group (SE3)
    * reflection을 제외한 rigid 변환을 proper rigid transformation이라 부르며, 3차원에서의 proper rigid transformation을 SE3라 함
    * 회전변환 R과 평행이동 t로 구성됨 (p' = R*p + t)
    * 오른손 좌표계를 기반으로 함 (엄지:x축, 검지:y축, 중지:z축)
    */

    class SE3
    {
    public:
        // 생성자
        SE3(void);                      // 기본생성자: 항등변환으로 초기화
        SE3(const cv::Matx34d& Rt);
        SE3(const cv::Mat& Rt);
        SE3(const cv::Matx33d& R, const cv::Matx31d& t);
        SE3(const cv::Mat& R, const cv::Mat& t);

        // 연산자
        inline SE3 operator * (const SE3& rhs) const { return SE3(m_R*rhs.m_R, m_R*rhs.m_t + m_t); }
        inline const SE3& operator *= (const SE3& rhs) { m_t=m_R*rhs.m_t + m_t; m_R=m_R*rhs.m_R; return *this; }

        void inv();
        inline const SE3 getInv() const { return SE3(m_R.t(), -m_R.t()*m_t); }	// m_R.t() == m_R.inv()

        void setIdentity();

        // 회전변환(R), 평행이동(t)
        const cv::Matx34d getRt() const;
        void getRt(cv::Matx34d& Rt) const;
        void setRt(const cv::Matx34d& Rt);
        void setRt(const cv::Mat& Rt);

        void getRt(cv::Matx33d& R, cv::Matx31d& t) const;
        void setRt(const cv::Matx33d& R, const cv::Matx31d& t);
        void setRt(const cv::Mat& R, const cv::Mat& t);

        // 회전변환(R) & 3D 방향(orientation)
        const cv::Matx33d& getRotation() const;
        void getRotation(cv::Matx33d& R) const;
        void setRotation(const cv::Matx33d& R);         // p' = m_R*p + m_t -> p' = R*p + m_t
        void setRotation(const cv::Mat& R);
        void setRotationPinned(const cv::Matx33d& R);   // p' = m_R*(p-m_R'*m_t) -> p' = R*(p-m_R'*m_t)
        void setRotationPinned(const cv::Mat& R);

        void getEulerAngles(double& pitch, double& roll, double& yaw) const;    // pitch:rx, roll:ry, yaw:rz
        void setEulerAngles(double pitch, double roll, double yaw);
        void setEulerAnglesPinned(double pitch, double roll, double yaw);

        void getEulerAnglesGraphics(double& pitch, double& roll, double& yaw) const;
        void setEulerAnglesGraphics(double pitch, double roll, double yaw);
        void setEulerAnglesGraphicsPinned(double pitch, double roll, double yaw);

        void getRodrigues(cv::Mat& rvec) const;	            // opencv rodrigues (theta = norm(rvec), v = rvec/theta)
        void setRodrigues(const cv::Mat& rvec);		        // opencv rodrigues (theta = norm(rvec), v = rvec/theta)
        void setRodriguesPinned(const cv::Mat& rvec);		// opencv rodrigues (theta = norm(rvec), v = rvec/theta)

        void getRodrigues(double& vx, double& vy, double& vz, double& theta_radian) const;
        void setRodrigues(double vx, double vy, double vz, double theta_radian);
        void setRodriguesPinned(double vx, double vy, double vz, double theta_radian);

        void getQuaternion(double& w, double& x, double& y, double& z) const;
        void setQuaternion(double w, double x, double y, double z);
        void setQuaternionPinned(double w, double x, double y, double z);

        // pan: 카메라 광학축이 world X축과 이루는 각 (반시계방향: +)
        // tilt: 카메라 광학축이 world XY평면과 이루는 각 (Z축 방향이 +)
        // roll: 광학축을 회전축으로 한 카메라의 회전각 (반시계방향: +)
        void getPanTiltRoll(double& pan, double& tilt, double& roll) const;
        void setPanTiltRoll(double pan, double tilt, double roll);
        void setPanTiltRollPinned(double pan, double tilt, double roll);

        // 평행이동(t) & 3D 위치(position)
        const cv::Matx31d& getTranslation() const;
        void setTranslation(const cv::Matx31d& t);
        void setTranslation(const cv::Mat& t);

        void getTranslation(double& tx, double& ty, double& tz) const;
        void setTranslation(double tx, double ty, double tz);

        void getPosition(double& x, double& y, double& z) const;
        void setPosition(double x, double y, double z);

        // 3D 자세(pose)
        void getPose(double& x, double& y, double& z, double& pitch, double& roll, double& yaw) const;      // Euler angles
        void setPose(double x, double y, double z, double pitch, double roll, double yaw);                  // Euler angles

        void getPoseGraphics(double& x, double& y, double& z, double& pitch, double& roll, double& yaw) const;
        void setPoseGraphics(double x, double y, double z, double pitch, double roll, double yaw);

        void getPosePanTiltRoll(double& x, double& y, double& z, double& pan, double& tilt, double& roll) const;
        void setPosePanTiltRoll(double x, double y, double z, double pan, double tilt, double roll);

        // utilities
        void display() const;
        static void display(const cv::Mat& m);

        static cv::Matx33d rotationMatX(double rad);    // X축 회전행렬을 생성하여 반환
        static cv::Matx33d rotationMatY(double rad);    // Y축 회전행렬을 생성하여 반환
        static cv::Matx33d rotationMatZ(double rad);    // Z축 회전행렬을 생성하여 반환

        static cv::Matx33d rotationFromEulerAngles(double pitch, double roll, double yaw);
        static cv::Matx33d rotationFromEulerAnglesGraphics(double pitch, double roll, double yaw);
        static cv::Matx33d rotationFromRodrigues(const cv::Mat& rvec);
        static cv::Matx33d rotationFromRodrigues(double vx, double vy, double vz, double theta_radian);
        static cv::Matx33d rotationFromQuaternion(double w, double x, double y, double z);
        static cv::Matx33d rotationFromPanTiltRoll(double pan, double tilt, double roll);

    protected:
        cv::Matx33d m_R;		// rotation (p' = m_R*p + m_t)
        cv::Matx31d m_t;		// translation (p' = m_R*p + m_t)
    };
} // End of 'evl'

#endif // End of '__EVL_SE3__'